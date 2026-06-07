use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use futures::StreamExt;
use glob::Pattern;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use reqwest::{Client, StatusCode};
use serde::Deserialize;
use tokio::io::AsyncWriteExt;
use tokio::sync::Semaphore;

const DEFAULT_INCLUDE: &str = "config.json,tokenizer.json,tokenizer_config.json,special_tokens_map.json,generation_config.json,*.safetensors,*.safetensors.index.json";

#[derive(Debug, Parser)]
#[command(
    name = "hf-downloader",
    about = "Parallel Hugging Face model downloader with resume and retry"
)]
struct Args {
    /// Model id in owner/repo form.
    repo: String,
    /// Output directory.
    out_dir: PathBuf,
    /// Explicit file list. Overrides --include when present.
    #[arg(long, num_args = 1..)]
    files: Vec<String>,
    /// Comma-separated glob include list.
    #[arg(long, default_value = DEFAULT_INCLUDE)]
    include: String,
    /// Hugging Face token. Prefer HF_TOKEN env so it does not enter shell history.
    #[arg(long, env = "HF_TOKEN")]
    token: Option<String>,
    /// Max in-flight file downloads.
    #[arg(long, default_value_t = 4)]
    concurrency: usize,
    /// Retry count per file.
    #[arg(long, default_value_t = 3)]
    retries: usize,
    /// Revision/branch/tag to resolve.
    #[arg(long, default_value = "main")]
    revision: String,
}

#[derive(Debug, Deserialize)]
struct ModelInfo {
    siblings: Vec<Sibling>,
}

#[derive(Debug, Clone, Deserialize)]
struct Sibling {
    rfilename: String,
    size: Option<u64>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    if !args.repo.contains('/') {
        bail!("expected owner/repo, got '{}'", args.repo);
    }
    if args.concurrency == 0 {
        bail!("--concurrency must be positive");
    }

    tokio::fs::create_dir_all(&args.out_dir)
        .await
        .with_context(|| format!("could not create {}", args.out_dir.display()))?;

    let client = Client::builder()
        .user_agent("tinygpt-hf-downloader/0.1")
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()?;

    let manifest = fetch_manifest(&client, &args.repo, args.token.as_deref()).await?;
    let files = select_files(&manifest.siblings, &args.files, &args.include)?;
    if files.is_empty() {
        bail!("no files selected for {}", args.repo);
    }

    eprintln!(
        "{}: {} file(s) -> {}",
        args.repo,
        files.len(),
        args.out_dir.display()
    );

    let mp = Arc::new(MultiProgress::new());
    let sem = Arc::new(Semaphore::new(args.concurrency));
    let mut tasks = Vec::new();
    for file in files {
        let permit = sem.clone().acquire_owned().await?;
        let client = client.clone();
        let repo = args.repo.clone();
        let out_dir = args.out_dir.clone();
        let token = args.token.clone();
        let revision = args.revision.clone();
        let mp = mp.clone();
        let retries = args.retries;
        tasks.push(tokio::spawn(async move {
            let _permit = permit;
            download_with_retry(
                &client,
                &repo,
                &revision,
                &file,
                &out_dir,
                token.as_deref(),
                retries,
                &mp,
            )
            .await
        }));
    }

    let mut ok = 0usize;
    for task in tasks {
        task.await??;
        ok += 1;
    }
    eprintln!("done: {ok} file(s)");
    Ok(())
}

async fn fetch_manifest(client: &Client, repo: &str, token: Option<&str>) -> Result<ModelInfo> {
    let url = format!("https://huggingface.co/api/models/{repo}");
    let mut req = client.get(&url);
    if let Some(token) = token.filter(|t| !t.is_empty()) {
        req = req.bearer_auth(token);
    }
    let resp = req.send().await?;
    match resp.status() {
        StatusCode::OK => Ok(resp.json::<ModelInfo>().await?),
        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
            bail!("{repo} requires HF_TOKEN or access approval")
        }
        StatusCode::NOT_FOUND => bail!("{repo} not found"),
        status => bail!("manifest request failed with HTTP {status}: {url}"),
    }
}

fn select_files(siblings: &[Sibling], explicit: &[String], include: &str) -> Result<Vec<Sibling>> {
    if !explicit.is_empty() {
        let mut selected = Vec::new();
        for name in explicit {
            match siblings.iter().find(|s| s.rfilename == *name) {
                Some(sibling) => selected.push(sibling.clone()),
                None => bail!("file '{}' not present in manifest", name),
            }
        }
        return Ok(selected);
    }

    let patterns: Vec<Pattern> = include
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(Pattern::new)
        .collect::<std::result::Result<_, _>>()?;
    Ok(siblings
        .iter()
        .filter(|sibling| patterns.iter().any(|p| p.matches(&sibling.rfilename)))
        .cloned()
        .collect())
}

async fn download_with_retry(
    client: &Client,
    repo: &str,
    revision: &str,
    file: &Sibling,
    out_dir: &Path,
    token: Option<&str>,
    retries: usize,
    mp: &MultiProgress,
) -> Result<()> {
    let mut attempt = 0usize;
    loop {
        match download_one(client, repo, revision, file, out_dir, token, mp).await {
            Ok(()) => return Ok(()),
            Err(err) if attempt < retries => {
                attempt += 1;
                let backoff = Duration::from_millis(500 * 2_u64.pow((attempt - 1) as u32));
                eprintln!(
                    "retry {attempt}/{retries} for {} after {backoff:?}: {err}",
                    file.rfilename
                );
                tokio::time::sleep(backoff).await;
            }
            Err(err) => return Err(err),
        }
    }
}

async fn download_one(
    client: &Client,
    repo: &str,
    revision: &str,
    file: &Sibling,
    out_dir: &Path,
    token: Option<&str>,
    mp: &MultiProgress,
) -> Result<()> {
    let dest = out_dir.join(&file.rfilename);
    if let Some(parent) = dest.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    if let Some(expected) = file.size {
        if let Ok(meta) = tokio::fs::metadata(&dest).await {
            if meta.len() == expected {
                eprintln!("cached: {}", file.rfilename);
                return Ok(());
            }
        }
    } else if tokio::fs::metadata(&dest).await.is_ok() {
        eprintln!("cached: {}", file.rfilename);
        return Ok(());
    }

    let url = hf_file_url(repo, revision, &file.rfilename);
    let pb = mp.add(ProgressBar::new(file.size.unwrap_or(0)));
    pb.set_style(progress_style());
    pb.set_message(file.rfilename.clone());
    if file.size.is_none() {
        pb.enable_steady_tick(Duration::from_millis(150));
    }

    let mut req = client.get(&url);
    if let Some(token) = token.filter(|t| !t.is_empty()) {
        req = req.bearer_auth(token);
    }
    let resp = req.send().await?;
    if resp.status() == StatusCode::UNAUTHORIZED || resp.status() == StatusCode::FORBIDDEN {
        bail!("{} requires HF_TOKEN or access approval", repo);
    }
    if !resp.status().is_success() {
        bail!("HTTP {} for {}", resp.status(), url);
    }

    let tmp = dest.with_extension(format!(
        "{}part",
        dest.extension()
            .and_then(|s| s.to_str())
            .map(|s| format!("{s}."))
            .unwrap_or_default()
    ));
    let mut out = tokio::fs::File::create(&tmp)
        .await
        .with_context(|| format!("could not create {}", tmp.display()))?;
    let mut stream = resp.bytes_stream();
    let mut written = 0u64;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        out.write_all(&chunk).await?;
        written += chunk.len() as u64;
        pb.set_position(written);
    }
    out.flush().await?;

    if let Some(expected) = file.size {
        if written != expected {
            let _ = tokio::fs::remove_file(&tmp).await;
            return Err(anyhow!(
                "{} size mismatch: got {}, expected {}",
                file.rfilename,
                written,
                expected
            ));
        }
    }

    tokio::fs::rename(&tmp, &dest)
        .await
        .with_context(|| format!("could not rename {} -> {}", tmp.display(), dest.display()))?;
    pb.finish_with_message(format!("{} done", file.rfilename));
    Ok(())
}

fn hf_file_url(repo: &str, revision: &str, filename: &str) -> String {
    let encoded = filename
        .split('/')
        .map(urlencoding::encode)
        .collect::<Vec<_>>()
        .join("/");
    format!("https://huggingface.co/{repo}/resolve/{revision}/{encoded}")
}

fn progress_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{msg:40} [{bar:30.cyan/blue}] {bytes}/{total_bytes} {bytes_per_sec} ETA {eta}",
    )
    .unwrap()
    .progress_chars("=> ")
}

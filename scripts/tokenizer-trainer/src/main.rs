use anyhow::{anyhow, Result};
use std::env;
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::normalizers::{Sequence, Strip, NFC};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::tokenizer::AddedToken;
use tokenizers::TokenizerBuilder;

#[derive(Debug)]
struct Config {
    corpus: String,
    vocab_size: usize,
    model_type: String,
    special_tokens: Vec<String>,
    out: String,
}

fn main() -> Result<()> {
    let cfg = parse_args()?;
    match cfg.model_type.as_str() {
        "bpe" => train_bpe(cfg),
        "char" => train_bpe(cfg),
        other => Err(anyhow!(
            "model-type '{}' is not supported by this v1 helper; use bpe or char",
            other
        )),
    }
}

fn parse_args() -> Result<Config> {
    let mut corpus = None;
    let mut vocab_size = 32_000usize;
    let mut model_type = "bpe".to_string();
    let mut special_tokens = vec!["<bos>".to_string(), "<eos>".to_string(), "<pad>".to_string()];
    let mut out = None;

    let args: Vec<String> = env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--corpus" => {
                corpus = args.get(i + 1).cloned();
                i += 2;
            }
            "--vocab-size" => {
                vocab_size = args
                    .get(i + 1)
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(vocab_size);
                i += 2;
            }
            "--model-type" => {
                model_type = args.get(i + 1).cloned().unwrap_or(model_type);
                i += 2;
            }
            "--special-tokens" => {
                special_tokens = args
                    .get(i + 1)
                    .map(|s| {
                        s.split(',')
                            .map(str::trim)
                            .filter(|s| !s.is_empty())
                            .map(str::to_string)
                            .collect()
                    })
                    .unwrap_or(special_tokens);
                i += 2;
            }
            "--out" => {
                out = args.get(i + 1).cloned();
                i += 2;
            }
            "-h" | "--help" => {
                usage();
                std::process::exit(0);
            }
            other => return Err(anyhow!("unknown flag '{}'", other)),
        }
    }

    Ok(Config {
        corpus: corpus.ok_or_else(|| anyhow!("--corpus required"))?,
        vocab_size,
        model_type,
        special_tokens,
        out: out.ok_or_else(|| anyhow!("--out required"))?,
    })
}

fn train_bpe(cfg: Config) -> Result<()> {
    let model = BPE::builder()
        .unk_token("[UNK]".to_string())
        .build()
        .map_err(|e| anyhow!("build BPE model: {}", e))?;
    let mut tokenizer = TokenizerBuilder::new()
        .with_model(model)
        .with_normalizer(Some(Sequence::new(vec![Strip::new(true, true).into(), NFC.into()])))
        .with_pre_tokenizer(Some(ByteLevel::default()))
        .with_post_processor(Some(ByteLevel::default()))
        .with_decoder(Some(ByteLevel::default()))
        .build()
        .map_err(|e| anyhow!("build tokenizer: {}", e))?;

    let added: Vec<AddedToken> = cfg
        .special_tokens
        .iter()
        .map(|s| AddedToken::from(s.as_str(), true))
        .collect();
    let mut trainer = BpeTrainerBuilder::new()
        .vocab_size(cfg.vocab_size)
        .special_tokens(added)
        .show_progress(true)
        .build();

    tokenizer
        .train_from_files(&mut trainer, vec![cfg.corpus.clone()])
        .map_err(|e| anyhow!("train tokenizer on {}: {}", cfg.corpus, e))?;
    tokenizer
        .save(&cfg.out, false)
        .map_err(|e| anyhow!("write {}: {}", cfg.out, e))?;
    eprintln!(
        "wrote tokenizer.json: {} (model_type={}, vocab_size={})",
        cfg.out, cfg.model_type, cfg.vocab_size
    );
    Ok(())
}

fn usage() {
    eprintln!(
        "usage: tinygpt-tokenizer-trainer --corpus corpus.txt --vocab-size 32000 --model-type bpe --out tokenizer.json"
    );
}

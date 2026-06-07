use std::fs;
use std::io::Write;
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use clap::Parser;
use serde_json::json;

#[derive(Debug, Parser)]
#[command(about = "Run generated Python code plus tests with macOS sandbox-exec and rlimits")]
struct Args {
    #[arg(long)]
    code: PathBuf,
    #[arg(long)]
    test: PathBuf,
    #[arg(long, default_value_t = 10)]
    timeout: u64,
    #[arg(long, default_value_t = 256)]
    memory_mb: u64,
    #[arg(long, default_value_t = false)]
    allow_network: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let started = Instant::now();
    let run_dir = std::env::temp_dir().join(format!(
        "tinygpt-humaneval-{}-{}",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis()
    ));
    fs::create_dir_all(&run_dir)?;
    let runner = run_dir.join("runner.py");
    let code = fs::read_to_string(&args.code)
        .with_context(|| format!("could not read {}", args.code.display()))?;
    let test = fs::read_to_string(&args.test)
        .with_context(|| format!("could not read {}", args.test.display()))?;
    let mut f = fs::File::create(&runner)?;
    writeln!(f, "{code}")?;
    writeln!(f, "\n{test}")?;
    writeln!(f, "\nif __name__ == '__main__':\n    check(candidate)")?;

    let profile = sandbox_profile(&run_dir, args.allow_network);
    let exe = if PathBuf::from("/usr/bin/sandbox-exec").exists() {
        "/usr/bin/sandbox-exec"
    } else {
        "/usr/bin/python3"
    };
    let mut cmd = Command::new(exe);
    if exe.ends_with("sandbox-exec") {
        cmd.args(["-p", &profile, "/usr/bin/python3"]);
    }
    cmd.arg(&runner)
        .current_dir(&run_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mem_bytes = args.memory_mb * 1024 * 1024;
    unsafe {
        cmd.pre_exec(move || {
            set_limit(libc::RLIMIT_CPU, args.timeout, args.timeout + 1)?;
            let _ = set_limit(libc::RLIMIT_AS, mem_bytes, mem_bytes);
            #[cfg(target_os = "macos")]
            {
                let _ = set_limit(libc::RLIMIT_DATA, mem_bytes, mem_bytes);
            }
            set_limit(libc::RLIMIT_NOFILE, 64, 64)?;
            Ok(())
        });
    }

    let mut child = cmd.spawn()?;
    let deadline = Instant::now() + Duration::from_secs(args.timeout + 1);
    let timed_out;
    loop {
        if child.try_wait()?.is_some() {
            timed_out = false;
            break;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            timed_out = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let output = child.wait_with_output()?;
    let wall = started.elapsed().as_secs_f64();
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let passed = output.status.success() && !timed_out;
    println!(
        "{}",
        json!({
            "passed": passed,
            "stdout": stdout,
            "stderr": if timed_out { format!("{stderr}\ntimeout") } else { stderr },
            "exit_code": output.status.code().unwrap_or(-1),
            "wall_seconds": wall
        })
    );
    let _ = fs::remove_dir_all(&run_dir);
    Ok(())
}

fn sandbox_profile(run_dir: &std::path::Path, allow_network: bool) -> String {
    let dir = run_dir.display();
    format!(
        r#"(version 1)
(deny default)
(allow process*)
(allow file-read*)
{}
(allow file-write* (subpath "{}"))
(allow sysctl-read)
"#,
        if allow_network {
            "(allow network*)"
        } else {
            "(deny network*)"
        },
        dir
    )
}

fn set_limit(resource: libc::c_int, soft: u64, hard: u64) -> std::io::Result<()> {
    let lim = libc::rlimit {
        rlim_cur: soft as libc::rlim_t,
        rlim_max: hard as libc::rlim_t,
    };
    let rc = unsafe { libc::setrlimit(resource, &lim) };
    if rc == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

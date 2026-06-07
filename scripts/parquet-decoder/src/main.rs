use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use arrow_array::{Array, RecordBatch, cast::AsArray};
use arrow_json::writer::{LineDelimited, WriterBuilder};
use arrow_schema::{ArrowError, DataType};
use clap::Parser;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

const FALLBACK_FIELDS: [&str; 3] = ["text", "content", "instruction"];

#[derive(Debug, Parser)]
#[command(
    name = "parquet-decoder",
    about = "Decode Hugging Face parquet shards to plain text or JSONL"
)]
struct Args {
    /// Parquet file or directory containing parquet shards.
    input: PathBuf,
    /// Output .txt or .jsonl path.
    output: PathBuf,
    /// Text column to extract in plain text mode.
    #[arg(long, default_value = "text")]
    field: String,
    /// Emit one JSON record per line instead of plain text.
    #[arg(long)]
    jsonl: bool,
    /// Cap total rows written.
    #[arg(long)]
    max_rows: Option<usize>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let shards = find_parquets(&args.input)?;
    if shards.is_empty() {
        bail!("no .parquet shards under {}", args.input.display());
    }

    eprintln!("[{} shard(s)] -> {}", shards.len(), args.output.display());
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("could not create {}", parent.display()))?;
    }

    let output = File::create(&args.output)
        .with_context(|| format!("could not create {}", args.output.display()))?;
    let mut out = BufWriter::new(output);
    let mut total = 0usize;
    let mut skipped = 0usize;

    for shard in shards {
        if args.max_rows.is_some_and(|max| total >= max) {
            break;
        }

        let file =
            File::open(&shard).with_context(|| format!("could not open {}", shard.display()))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .with_context(|| format!("could not read {}", shard.display()))?;
        let schema = builder.schema().clone();
        let cols: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
        let picked = pick_field(&args.field, &cols);
        let row_count = builder.metadata().file_metadata().num_rows();

        eprintln!(
            "  {}: {} rows, columns={:?}, picking '{}'",
            shard
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("<unknown>"),
            row_count,
            cols,
            picked.unwrap_or("<none>")
        );

        let reader = builder.with_batch_size(8192).build()?;
        if args.jsonl {
            write_jsonl(reader, &mut out, args.max_rows, &mut total)?;
        } else if let Some(field) = picked {
            write_text(
                reader,
                &mut out,
                field,
                args.max_rows,
                &mut total,
                &mut skipped,
            )?;
        } else {
            eprintln!("  ! no text-ish column in {}; skipping", shard.display());
        }
    }

    out.flush()?;
    let size = args.output.metadata()?.len();
    if skipped == 0 {
        eprintln!(
            "wrote {} rows ({} bytes) to {}",
            total,
            size,
            args.output.display()
        );
    } else {
        eprintln!(
            "wrote {} rows ({} bytes) to {}; skipped {} empty",
            total,
            size,
            args.output.display(),
            skipped
        );
    }

    Ok(())
}

fn find_parquets(path: &Path) -> Result<Vec<PathBuf>> {
    if path.is_file() {
        return if path.extension().and_then(|ext| ext.to_str()) == Some("parquet") {
            Ok(vec![path.to_path_buf()])
        } else {
            bail!("input file is not .parquet: {}", path.display());
        };
    }

    if !path.is_dir() {
        bail!("no parquet at {}", path.display());
    }

    let mut shards = Vec::new();
    collect_parquets(path, &mut shards)?;
    shards.sort();
    Ok(shards)
}

fn collect_parquets(path: &Path, shards: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(path).with_context(|| format!("could not read {}", path.display()))? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_parquets(&path, shards)?;
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("parquet") {
            shards.push(path);
        }
    }
    Ok(())
}

fn pick_field<'a>(requested: &'a str, cols: &'a [String]) -> Option<&'a str> {
    if cols.iter().any(|col| col == requested) {
        return Some(requested);
    }
    FALLBACK_FIELDS
        .iter()
        .copied()
        .find(|fallback| cols.iter().any(|col| col == fallback))
}

fn write_jsonl<I, W>(
    reader: I,
    out: &mut W,
    max_rows: Option<usize>,
    total: &mut usize,
) -> Result<()>
where
    I: IntoIterator<Item = std::result::Result<RecordBatch, ArrowError>>,
    W: Write,
{
    let mut writer = WriterBuilder::new()
        .with_explicit_nulls(true)
        .build::<_, LineDelimited>(out);

    for batch in reader {
        let batch = batch?;
        let batch = cap_batch(batch, max_rows, *total);
        if batch.num_rows() == 0 {
            break;
        }
        *total += batch.num_rows();
        writer.write(&batch)?;
        if max_rows.is_some_and(|max| *total >= max) {
            eprintln!("  ! hit --max-rows {}, stopping", max_rows.unwrap());
            break;
        }
    }
    writer.finish()?;
    Ok(())
}

fn cap_batch(batch: RecordBatch, max_rows: Option<usize>, total: usize) -> RecordBatch {
    match max_rows {
        Some(max) if total + batch.num_rows() > max => batch.slice(0, max.saturating_sub(total)),
        _ => batch,
    }
}

fn write_text<I, W>(
    reader: I,
    out: &mut W,
    field: &str,
    max_rows: Option<usize>,
    total: &mut usize,
    skipped: &mut usize,
) -> Result<()>
where
    I: IntoIterator<Item = std::result::Result<RecordBatch, ArrowError>>,
    W: Write,
{
    for batch in reader {
        let batch = batch?;
        let Some(array) = batch.column_by_name(field) else {
            continue;
        };

        for row in 0..array.len() {
            if max_rows.is_some_and(|max| *total >= max) {
                eprintln!("  ! hit --max-rows {}, stopping", max_rows.unwrap());
                return Ok(());
            }
            if array.is_null(row) {
                *skipped += 1;
                continue;
            }
            let Some(value) = text_value(array.as_ref(), row) else {
                *skipped += 1;
                continue;
            };
            if value.is_empty() {
                *skipped += 1;
                continue;
            }
            out.write_all(value.as_bytes())?;
            out.write_all(b"\n\n")?;
            *total += 1;
        }
    }
    Ok(())
}

fn text_value(array: &dyn Array, row: usize) -> Option<String> {
    match array.data_type() {
        DataType::Utf8 => Some(array.as_string::<i32>().value(row).to_owned()),
        DataType::LargeUtf8 => Some(array.as_string::<i64>().value(row).to_owned()),
        DataType::Utf8View => Some(array.as_string_view().value(row).to_owned()),
        _ => None,
    }
}

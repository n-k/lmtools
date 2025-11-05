use std::time::Duration;

use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::{params, OptionalExtension};

pub async fn dir_scanner(pool: Pool<SqliteConnectionManager>) -> anyhow::Result<()> {
    loop {
        tokio::time::sleep(Duration::from_millis(100)).await;
        let conn = pool.get()?;

        let mut stmt =
            conn.prepare("SELECT id, path FROM dir_queue WHERE status = 'pending' LIMIT 1")?;

        let mut rows = stmt.query([])?;

        let Some(row) = rows.next()? else {
            continue;
        };
        let id: i64 = row.get(0)?;
        let path: String = row.get(1)?;

        conn.execute("UPDATE dir_queue SET status='scanning' WHERE id=?", [id])
            .unwrap();
        conn.cache_flush()?;

        for entry in std::fs::read_dir(path).unwrap() {
            let entry = entry.unwrap();
            let p = entry.path();
            let entry_path = p.as_os_str().to_str().unwrap();
            let is_symlink = entry
                .metadata()
                .map(|md| md.is_symlink())
                .unwrap_or_else(|_| false);
            if is_symlink {
                continue;
            }
            if p.is_dir() {
                conn.execute(
                    "INSERT OR IGNORE INTO dir_queue (path) VALUES (?);",
                    [entry_path],
                )?;
            } else {
                conn.execute(
                    "INSERT OR IGNORE INTO file_queue (dir_id, path) VALUES (?1, ?2);",
                    params![id, entry_path],
                )?;
            }
        }
        conn.execute("UPDATE dir_queue SET status='done' WHERE id=?", [id])
            .unwrap();
    }
}

pub async fn file_scanner(pool: Pool<SqliteConnectionManager>) -> anyhow::Result<()> {
    loop {
        tokio::time::sleep(Duration::from_millis(100)).await;
        let conn = pool.get()?;
        let Some((id, path)): Option<(i64, String)> = conn
            .query_one(
                "SELECT id, path FROM file_queue WHERE status = 'pending' LIMIT 1",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?
        else {
            continue;
        };

        // Update status to scanning
        conn.execute("UPDATE file_queue SET status='scanning' WHERE id=?", [id])?;
        conn.cache_flush()?;

        // Check if it's a text file
        if !is_text_file(&path) {
            conn.execute("UPDATE file_queue SET status='done' WHERE id=?", [id])?;
            continue;
        }

        // Read and process the file
        match std::fs::read_to_string(&path) {
            Ok(content) => {
                // Chunk the content
                let chunks = chunk_text(&content, 2000);

                // Insert chunks into documents table
                for (chunk_index, chunk) in chunks.into_iter().enumerate() {
                    conn.execute(
                        "INSERT INTO documents (file_path, chunk_index, content) VALUES (?, ?, ?)",
                        params![&path, chunk_index as i64, chunk],
                    )?;
                }

                conn.execute("UPDATE file_queue SET status='done' WHERE id=?", [id])?;
            }
            Err(e) => {
                // Mark as error if reading fails
                conn.execute(
                    "UPDATE file_queue SET status='error', error=? WHERE id=?",
                    params![format!("Failed to read file: {}", e), id],
                )?;
            }
        }
    }
}

fn is_text_file(path: &str) -> bool {
    let text_extensions = [
        "txt",
        "rs",
        "js",
        "ts",
        "py",
        "java",
        "c",
        "cpp",
        "h",
        "hpp",
        "html",
        "css",
        "md",
        "json",
        "yaml",
        "yml",
        "toml",
        "xml",
        "sh",
        "bash",
        "zsh",
        "fish",
        "sql",
        "go",
        "rb",
        "php",
        "pl",
        "pm",
        "lua",
        "r",
        "scala",
        "kt",
        "swift",
        "dart",
        "hs",
        "ml",
        "fs",
        "fsx",
        "vb",
        "cs",
        "fs",
        "elm",
        "clj",
        "cljs",
        "ex",
        "exs",
        "nim",
        "cr",
        "v",
        "zig",
        "jl",
        "scm",
        "ss",
        "rkt",
        "hy",
        "coffee",
        "litcoffee",
        "ls",
        "moon",
        "iced",
        "pl",
        "pm",
        "t",
        "pod",
        "awk",
        "sed",
        "makefile",
        "dockerfile",
        "gitignore",
        "readme",
        "license",
        "changelog",
        "authors",
        "contributors",
        "news",
        "history",
        "todo",
    ];

    if let Some(ext) = std::path::Path::new(path).extension() {
        if let Some(ext_str) = ext.to_str() {
            return text_extensions.contains(&ext_str.to_lowercase().as_str());
        }
    }

    // For files without extensions, try to read first few bytes to check if text
    if let Ok(bytes) = std::fs::read(&path) {
        if bytes.len() > 0 {
            // Check if first 1024 bytes are valid UTF-8
            let check_len = std::cmp::min(1024, bytes.len());
            return std::str::from_utf8(&bytes[..check_len]).is_ok();
        }
    }

    false
}

fn chunk_text(text: &str, max_chunk_size: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut start = 0;

    while start < text.len() {
        let end = std::cmp::min(start + max_chunk_size, text.len());

        // Try to break at word boundaries if possible
        let chunk_end = if end < text.len() {
            // Look for last whitespace within the last 100 characters
            let search_start = if end > 100 { end - 100 } else { start };
            if let Some(last_space) = text[search_start..end].rfind(char::is_whitespace) {
                search_start + last_space + 1
            } else {
                end
            }
        } else {
            end
        };

        chunks.push(text[start..chunk_end].to_string());
        start = chunk_end;
    }

    chunks
}

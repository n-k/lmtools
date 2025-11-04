use std::time::Duration;

use r2d2::PooledConnection;
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::params;

pub async fn dir_scanner(
    conn: PooledConnection<SqliteConnectionManager>
) -> anyhow::Result<()> {
    // let conn = pool.get()?;
    loop {
        tokio::time::sleep(Duration::from_millis(100)).await;
        
        let mut stmt = conn.prepare(
            "SELECT id, path FROM dir_queue WHERE status = 'pending' LIMIT 1"
        )?;

        let mut rows = stmt.query([])?;

        let Some(row) = rows.next()? else { continue; };
        let id: i64 = row.get(0)?;
        let path: String = row.get(1)?;

        conn.execute("UPDATE dir_queue SET status='scanning' WHERE id=?", [id]).unwrap();
        conn.cache_flush()?;
        
        for entry in std::fs::read_dir(path).unwrap() {
            let entry = entry.unwrap();
            let p = entry.path();
            let entry_path = p.as_os_str().to_str().unwrap();
            if p.is_dir() {
                conn.execute(
                    "INSERT OR IGNORE INTO dir_queue (path) VALUES (?);", 
                    [entry_path]
                )?;
                // enqueue_dir(&conn, &p);
            } else {
                conn.execute(
                    "INSERT OR IGNORE INTO file_queue (dir_id, path) VALUES (?1, ?2);", 
                    params![id, entry_path]
                )?;
            }
        }
        conn.execute("UPDATE dir_queue SET status='done' WHERE id=?", [id]).unwrap();
    }
}

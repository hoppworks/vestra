use clap::{Parser, Subcommand};
use vestra_core::{WindowSettings, plan_windows};

#[derive(Debug, Parser)]
#[command(name = "vestra", about = "Native video-to-world reconstruction")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Print the deterministic multi-view window schedule.
    Plan {
        #[arg(long)]
        frames: usize,
        #[arg(long, default_value_t = 12)]
        chunk_size: usize,
        #[arg(long, default_value_t = 3)]
        overlap: usize,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command {
        Command::Plan {
            frames,
            chunk_size,
            overlap,
        } => {
            let windows = plan_windows(
                frames,
                WindowSettings {
                    chunk_size,
                    overlap,
                },
            )?;
            println!("{}", serde_json::to_string_pretty(&windows)?);
        }
    }
    Ok(())
}

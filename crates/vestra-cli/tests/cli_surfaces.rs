use std::process::{Command, Output};

const PRODUCT_COMMANDS: [&str; 6] = ["app", "reconstruct", "demo", "serve", "inspect", "export"];
const LAB_COMMANDS: [&str; 28] = [
    "plan",
    "fuse",
    "pose-import-colmap",
    "pose-import-colmap-model",
    "pose-import-json",
    "fuse-colmap-global",
    "fuse-global-pose",
    "inspect-colmap-global",
    "inspect-global-pose",
    "inspect-colmap-frame-global",
    "fuse-colmap-frame-global",
    "import-colmap-mvs",
    "import-da3-pose-conditioned",
    "fuse-da3-pose-conditioned-tsdf",
    "extract-architecture",
    "raster-record",
    "export-glb",
    "export-splat",
    "export-cameras",
    "verify",
    "oracle-fixture",
    "oracle-model-bench",
    "oracle-inspect",
    "oracle-stitch",
    "oracle-compare",
    "oracle-compare-capi",
    "oracle-compare-model",
    "oracle-run",
];

fn run(executable: &str, arguments: &[&str]) -> Output {
    Command::new(executable)
        .args(arguments)
        .output()
        .expect("CLI binary should run")
}

fn visible_commands(output: &Output) -> Vec<String> {
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout.clone()).expect("help should be UTF-8");
    let commands = stdout
        .split_once("Commands:\n")
        .expect("help should have a command section")
        .1
        .split_once("\n\nOptions:")
        .expect("help should have an options section")
        .0;
    commands
        .lines()
        .filter_map(|line| line.split_whitespace().next())
        .filter(|name| *name != "help")
        .map(str::to_owned)
        .collect()
}

#[test]
fn product_binary_help_exposes_exactly_six_commands() {
    let output = run(env!("CARGO_BIN_EXE_vestra"), &["--help"]);
    assert_eq!(visible_commands(&output), PRODUCT_COMMANDS);
}

#[test]
fn lab_binary_help_exposes_exactly_the_other_twenty_eight_commands() {
    let output = run(env!("CARGO_BIN_EXE_vestra-lab"), &["--help"]);
    assert_eq!(visible_commands(&output), LAB_COMMANDS);
}

#[test]
fn product_binary_rejects_a_lab_command_without_a_hidden_alias() {
    let output = run(env!("CARGO_BIN_EXE_vestra"), &["plan"]);
    assert_eq!(output.status.code(), Some(2));
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("unrecognized subcommand 'plan'"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn lab_binary_rejects_a_product_command_without_a_hidden_alias() {
    let output = run(env!("CARGO_BIN_EXE_vestra-lab"), &["reconstruct"]);
    assert_eq!(output.status.code(), Some(2));
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("unrecognized subcommand 'reconstruct'"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

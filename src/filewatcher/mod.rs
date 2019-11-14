use notify::{DebouncedEvent, RecommendedWatcher, RecursiveMode, Watcher};
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver};
use std::thread;
use std::time::Duration;

pub fn watch_path(path: PathBuf) -> Receiver<DebouncedEvent> {
    let (sender, recv) = channel();
    let _child = thread::spawn(move || {
        let (tx, rx) = channel();

        let mut shader_watcher: RecommendedWatcher =
            Watcher::new(tx, Duration::from_millis(50)).unwrap();
        loop {
            shader_watcher
                .watch(path.clone(), RecursiveMode::Recursive)
                .unwrap();
            let event = rx.recv().unwrap();
            sender.send(event).unwrap();
        }
    });

    recv
}

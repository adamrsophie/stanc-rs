//! Where the output of Stan's `print` statements goes.
//!
//! Stan passes an output stream to every model method. Here the stream belongs to the thread
//! instead, so the methods keep their signatures: a caller that runs a model on a thread sets
//! that thread's sink, and chains on separate threads keep separate outputs. Without a sink,
//! printed lines go to standard output, as with CmdStan.

use std::cell::RefCell;
use std::io::Write;

thread_local! {
    static SINK: RefCell<Option<Box<dyn Write>>> = const { RefCell::new(None) };
}

/// Sends this thread's printed lines to `sink` (standard output for `None`) and returns the
/// previous sink, so that a caller can restore it or read back what a buffer collected.
pub fn set_sink(sink: Option<Box<dyn Write>>) -> Option<Box<dyn Write>> {
    SINK.with(|s| std::mem::replace(&mut *s.borrow_mut(), sink))
}

/// Runs `f` with this thread's printed lines going to `sink`, then restores the previous sink
/// and returns `f`'s result together with `sink`.
pub fn with_sink<W: Write + 'static, R>(sink: W, f: impl FnOnce() -> R) -> (R, W) {
    struct Restore(Option<Option<Box<dyn Write>>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            if let Some(prev) = self.0.take() {
                set_sink(prev);
            }
        }
    }
    let shared = std::rc::Rc::new(RefCell::new(Some(sink)));
    let mut restore = Restore(Some(set_sink(Some(Box::new(Shared(shared.clone()))))));
    let r = f();
    if let Some(prev) = restore.0.take() {
        set_sink(prev);
    }
    let sink = shared.borrow_mut().take().expect("sink taken once");
    (r, sink)
}

/// A writer the thread-local slot can own while [`with_sink`] keeps access to it.
struct Shared<W>(std::rc::Rc<RefCell<Option<W>>>);

impl<W: Write> Write for Shared<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self.0.borrow_mut().as_mut() {
            Some(w) => w.write(buf),
            None => Ok(buf.len()),
        }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        match self.0.borrow_mut().as_mut() {
            Some(w) => w.flush(),
            None => Ok(()),
        }
    }
}

/// Writes one printed line. Output errors are ignored, as a failed `print` must not change
/// the model's result.
pub fn line(s: &str) {
    SINK.with(|sink| match sink.borrow_mut().as_mut() {
        Some(w) => {
            let _ = writeln!(w, "{s}");
        }
        None => println!("{s}"),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captures_and_restores() {
        let ((), buf) = with_sink(Vec::new(), || {
            line("a");
            line("b");
        });
        assert_eq!(buf, b"a\nb\n");
        let ((), outer) = with_sink(Vec::new(), || {
            let ((), inner) = with_sink(Vec::new(), || line("in"));
            assert_eq!(inner, b"in\n");
            line("out");
        });
        assert_eq!(outer, b"out\n");
    }
}

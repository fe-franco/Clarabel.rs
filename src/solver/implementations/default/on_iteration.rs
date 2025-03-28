use crate::solver::traits::{CloneableFnMut, IterationCallback};

///  Implement Clone for IterationCallback using clone_box.
impl<T> Clone for IterationCallback<T> {
    fn clone(&self) -> Self {
        IterationCallback(self.0.as_ref().map(|callback| callback.clone_box()))
    }
}

impl<T> std::fmt::Debug for IterationCallback<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IterationCallback").finish()
    }
}

// Implement clone_box for the CloneableFnMut trait.
impl<T, F> CloneableFnMut<T> for F
where
    F: FnMut((&[T], &[T], &[T]), u32) -> Result<(), String> + Clone + Send + Sync + 'static,
{
    fn clone_box(&self) -> Box<dyn CloneableFnMut<T>> {
        Box::new(self.clone())
    }
}

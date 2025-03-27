use crate::solver::IterationCallback;
use pyo3::prelude::*;
use std::fmt::Debug;

/// A wrapper around a Python object that implements Clone.
#[derive(Debug)]
pub struct PyObjCloneable(PyObject);

impl Clone for PyObjCloneable {
    /// Clone the Python object by increasing its refcount.
    fn clone(&self) -> Self {
        Python::with_gil(|py| PyObjCloneable(self.0.clone_ref(py)))
    }
}

/// Convert a Python callback into a Rust `IterationCallback<f64>` without using Arc.
pub fn python_on_iteration_to_rust(py_on_iteration: PyObject) -> IterationCallback<f64> {
    // Wrap the PyObject in our custom cloneable wrapper.
    let py_on_iteration = PyObjCloneable(py_on_iteration);
    IterationCallback(Some(Box::new(move |x, iter| {
        // Clone the wrapper; this safely increases the Python object's refcount.
        let py_on_iteration = py_on_iteration.clone().0;
        Python::with_gil(|py| {
            // Convert the slice to a Python list using into_pyobject and handle errors.
            let x_py = match x.to_vec().into_pyobject(py) {
                Ok(obj) => obj,
                Err(err) => {
                    err.print(py);
                    return Err("Failed to convert Rust slice to Python object".to_string());
                }
            };
            // Build the arguments tuple: (x, iteration)
            let args = (x_py, iter);
            // Call the Python on_iteration and ignore its return value.
            if let Err(e) = py_on_iteration.call1(py, args) {
                e.print(py);
                return Err(e.to_string());
            }
            Ok(())
        })
    })))
}

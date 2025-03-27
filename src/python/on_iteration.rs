use crate::solver::IterationCallback;
use pyo3::conversion::IntoPyObject;
use pyo3::prelude::*;

/// A wrapper around PyObject that implements Clone by calling `clone_ref` with the GIL.
/// We derive FromPyObject and IntoPyObject so that conversion is automatic.
#[derive(Debug, pyo3::FromPyObject, pyo3::IntoPyObject)]
pub struct PyObjCloneable(pub PyObject);

impl Clone for PyObjCloneable {
    fn clone(&self) -> Self {
        Python::with_gil(|py| PyObjCloneable(self.0.clone_ref(py)))
    }
}

/// Convert a Python on_iteration callback (a PyObject) into a Rust IterationCallback<f64>
/// without using Arc.
pub fn python_on_iteration_to_rust(py_on_iteration: PyObject) -> IterationCallback<f64> {
    // Wrap the provided Python callback in our custom cloneable type.
    let py_on_iteration = PyObjCloneable(py_on_iteration);
    IterationCallback(Some(Box::new(move |variables, iter| {
        // Clone our wrapper so that the Python callback's refcount increases safely.
        let py_on_iteration = py_on_iteration.clone().0;
        Python::with_gil(|py| {
            // Destructure the tuple (x, s, z) from the solver.
            let (x, s, z) = variables;
            // Convert the entire tuple (x.to_vec(), s.to_vec(), z.to_vec()) at once.
            let variables_py: PyObject = (x.to_vec(), s.to_vec(), z.to_vec())
                .into_pyobject(py)
                .map_err(|_| "Failed to convert on_iteration variables to python".to_string())?
                .into();
            // Build the arguments tuple for the Python callback.
            let args = (variables_py, iter);
            // Call the Python callback and propagate errors as strings.
            py_on_iteration
                .call1(py, args)
                .map(|_| ())
                .map_err(|e| e.to_string())
        })
    })))
}

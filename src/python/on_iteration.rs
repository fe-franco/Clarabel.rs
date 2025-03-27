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
    let py_on_iteration = PyObjCloneable(py_on_iteration);
    IterationCallback(Some(Box::new(move |x, iter| {
        // Clone our wrapper so that the Python callback's refcount is increased safely.
        let py_on_iteration = py_on_iteration.clone().0;
        Python::with_gil(|py| {
            // Convert the Rust Vec<f64> into a Python object.
            // The conversion returns a Bound smart pointer, we convert it into a PyObject.
            let x_py: PyObject = x.to_vec().into_pyobject(py).unwrap().into();
            let args = (x_py, iter);
            match py_on_iteration.call1(py, args) {
                Ok(_) => Ok(()),
                Err(e) => Err(e.to_string()),
            }
        })
    })))
}

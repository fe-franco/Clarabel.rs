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
    IterationCallback(Some(Box::new(move |variables, iter| {
        // Clone our wrapper so that the Python callback's refcount is increased safely.
        let py_on_iteration = py_on_iteration.clone().0;
        Python::with_gil(|py| {
            // Convert the Rust (Vec<f64>,Vec<f64>,Vec<f64>) into a Python object.
            // The conversion returns a Bound smart pointer, we convert it into a PyObject.
            let (x, s, z) = variables;

            let x_py: PyObject = x.to_vec().into_pyobject(py).unwrap().into();
            let s_py: PyObject = s.to_vec().into_pyobject(py).unwrap().into();
            let z_py: PyObject = z.to_vec().into_pyobject(py).unwrap().into();

            let variables_py = (x_py, s_py, z_py);

            let args = (variables_py, iter);
            match py_on_iteration.call1(py, args) {
                Ok(_) => Ok(()),
                Err(e) => Err(e.to_string()),
            }
        })
    })))
}

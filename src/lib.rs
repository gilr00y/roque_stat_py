use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{array, PyArray, PyArray1, PyArray2};
use std::collections::HashMap;
use numpy::ndarray::{Array1, Array2};
use roque_stat::roque_stat::batch_crp::BatchCRP;
use roque_stat::roque_stat::crp::CRP;
use roque_stat::roque_stat::table::Table;

/// Formats the sum of two numbers as string.
// #[pyfunction]
// fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
//     Ok((a + b).to_string())
// }
//
// /// A Python module implemented in Rust.
// #[pymodule]
// fn roque_stat(_py: Python, m: &PyModule) -> PyResult<()> {
//     m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
//     Ok(())
// }

#[pyclass]
pub struct PyBatchCRP {
  batch_crp: BatchCRP<'static>,
}

// Implement necessary methods for PyBatchCRP

#[pymodule]
fn proque_stat(_py: Python, m: &PyModule) -> PyResult<()> {
  m.add_class::<PyBatchCRP>()?;

  Ok(())
}

#[pymethods]
impl PyBatchCRP {
  // Example: new() method for Python
  #[new]
  pub unsafe fn new(alpha: f64, max_iterations: u32, psi_scale: &PyArray1<f64>) -> Self {
    let psi_scale: Array1<f64> = psi_scale.as_array().to_owned();
    let tables = Box::leak(Box::new(HashMap::<Vec<u8>, Box<Table>>::new()));
    let batch_crp = BatchCRP {
      alpha,
      max_iterations,
      tables,
      psi_scale,
    };
    PyBatchCRP { batch_crp }
  }

  // Example: seat() method for Python
  pub unsafe fn seat(&mut self, datum: &PyArray1<f64>) {
    let datum: Array1<f64> = datum.as_array().to_owned();
    self.batch_crp.seat(datum);
  }

  // Implement other necessary methods similarly
  pub fn draw(&self) -> PyResult<Py<PyArray1<f64>>> {
    let samples = self.batch_crp.draw(1);
    let first_sample: &Array1<f64> = samples.first().unwrap();

    Python::with_gil(|py| {
      let py_array: Py<PyArray1<f64>> = PyArray::from_array(py, first_sample).to_owned();
      Ok(py_array)
    })
  }

  pub unsafe fn pp(&self, datum: &PyArray1<f64>) -> f64 {
    self.batch_crp.pp(datum.to_owned_array())
  }
}


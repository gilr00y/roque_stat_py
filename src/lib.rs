use pyo3::prelude::*;
use pyo3::types::{PyDict, PyInt};
use numpy::{array, PyArray, PyArray1, PyArray2};
use std::collections::HashMap;
use numpy::ndarray::{Array1, Array2};
use roque_stat::roque_stat::batch_crp::BatchCRP;
use roque_stat::roque_stat::conditional::{Conditional, ConditionalQuery};
use roque_stat::roque_stat::crp::CRP;
use roque_stat::roque_stat::projection::Projection;
use roque_stat::roque_stat::kl::KLDivergence;

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
pub struct Roque {
  batch_crp: BatchCRP,
}

// Implement necessary methods for PyBatchCRP

#[pymodule]
fn proque_stat(_py: Python, m: &PyModule) -> PyResult<()> {
  m.add_class::<Roque>()?;

  Ok(())
}

#[pymethods]
impl Roque {
  // Example: new() method for Python
  #[new]
  pub unsafe fn new(alpha: f64, max_iterations: u32, psi_scale: &PyArray1<f64>) -> Self {
    let psi_scale: Array1<f64> = psi_scale.as_array().to_owned();
    let tables = Box::new(HashMap::<Vec<u8>, Box<Table>>::new());
    let batch_crp = BatchCRP {
      alpha,
      max_iterations,
      tables,
      psi_scale,
    };
    Roque { batch_crp }
  }

  pub unsafe fn project(&self, projection: &PyArray1<i64>) -> Self {
    let proj_conv: Vec<usize> = projection.as_array().to_vec().iter().map(|x| usize::try_from(x.clone()).unwrap()).collect();
    Roque {
      batch_crp: Projection::new(&self.batch_crp, proj_conv).crp
    }
  }

  pub unsafe fn query(&self, query: &PyDict) -> Self {
    let query_conv: ConditionalQuery = query.extract().unwrap();
    let conditional = Conditional::new(&self.batch_crp, query_conv);
    Roque {
      batch_crp: conditional.crp
    }
  }

  pub unsafe fn compare(&self, other: &Roque) -> (f64, f64, f64) {
    let kl = KLDivergence::new(&self.batch_crp, &other.batch_crp);
    (kl.upper, kl.lower, kl.mean)
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

  pub fn describe(&self) -> PyResult<PyObject> {
    let mut map: HashMap<&str, &str> = HashMap::new();
    let table_count = self.batch_crp.tables.len().to_string();
    let binding = table_count.as_str();
    map.insert("num_tables", binding);

    pyo3::Python::with_gil(|py| {
      Ok(map.to_object(py))
    })
  }
}

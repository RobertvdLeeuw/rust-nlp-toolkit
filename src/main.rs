use std::collections::HashMap;
use std::iter::FromIterator;
use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

#[derive(Debug)]
enum Tensor {
    Scalar(f32),
    NTensor(Vec<Tensor>, usize)
}


impl Index<usize> for Tensor {
    type Output = Tensor;

    fn index(&self, index: usize) -> &Tensor {
        match self {
            Tensor::Scalar(_) => panic!("Attempted to index a scalar value"),
            Tensor::NTensor(vec, _) => &vec[index],
        }
    }
}

impl IndexMut<usize> for Tensor {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match self {
            Tensor::Scalar(_) => panic!("Attempted to index a scalar value"),
            Tensor::NTensor(vec, _) => &mut vec[index],
        }
    }
}


impl FromIterator<Tensor> for Tensor {
    fn from_iter<I: IntoIterator<Item = Tensor>>(iter: I) -> Self {
        let vec: Vec<Tensor> = iter.into_iter().collect();

        match vec[0].clone() {
            Tensor::Scalar(_) => Tensor::NTensor(vec, 1),
            Tensor::NTensor(_, n_dim) => Tensor::NTensor(vec, n_dim+1)
        }
    }
}


impl Tensor {
    fn clone(&self) -> Tensor {
        match self {
            Tensor::Scalar(val) => Tensor::Scalar(val.clone()),
            Tensor::NTensor(ten, n_dim) => ten.iter()
                                              .map(|t| t.clone())
                                              .collect()
        }
    }

    fn shape(&self) -> Vec<usize> {
        match self {
            Tensor::Scalar(_) => vec![],
            Tensor::NTensor(vec, _) => {
                let mut s = vec![vec.len()];
                s.extend(vec[0].shape());

                s
            }
        }
    }

        fn get_size(&self, dim: usize) -> usize {
        match self {
            Tensor::Scalar(_) => 1,
            Tensor::NTensor(ten, _) => {
                if dim == 0 {
                    ten.len()
                } else {
                    ten[0].get_size(dim - 1)
                }
            }
        }
    }

    fn get_element(&self, dim: usize, index: usize) -> Tensor {
        match self {
            Tensor::Scalar(_) => self.clone(),
            Tensor::NTensor(ten, _) => {
                if dim == 0 {
                    ten[index].clone()
                } else {
                    ten[0].get_element(dim - 1, index)
                }
            }
        }
    }

    fn get_slice(&self, dim: usize, index: usize) -> Tensor {
        match self {
            Tensor::Scalar(_) => self.clone(),
            Tensor::NTensor(ten, n_dim) => {
                if dim == 0 {
                    ten[index].clone()
                } else {
                    Tensor::NTensor(
                        ten.iter().map(|t| t.get_slice(dim - 1, index)).collect(),
                        *n_dim
                    )
                }
            }
        }
    }

    fn identity(size: usize, dimensions: usize) -> Tensor {
        if size <= 0 {
            panic!("Size must be > 0");
        }

        if dimensions == 0 {
            return Tensor::Scalar(1.0);
        }


        fn recursive_traverse(td: &Tensor, i: usize) -> Tensor {
            match td.clone() {
                Tensor::NTensor(mut vec, n_dim) => {
                    match vec[i] {
                        Tensor::Scalar(_) => {
                            vec[i] = Tensor::Scalar(1.0);
                        },
                        Tensor::NTensor(_, _) => {
                            vec[i] = recursive_traverse(&vec[i], i);
                        },
                    }
                    
                    Tensor::NTensor(vec, n_dim)
                },
                _ => panic!("Recursive traverse error, write this later!"),
            }
        }
        
        Ok(Tensor::filled(0.0, vec![size; dimensions]).iter()
                                                      .zip()
                                                      .map(|(t)| t[i] = recursive_traverse(&t[i], i))
                                                      .into::<Tensor>())
    }

    fn filled(value: f32, dimensions: Vec<usize>) -> Tensor {
        if dimensions.is_empty() {
            return Tensor::Scalar(value);
        }

        Tensor::NTensor(vec![Tensor::filled(value, dimensions[1..].to_vec()); 
                             dimensions[0]], 
                        dimensions.len())
    }

    fn transpose_swap(&self, mut a: usize, mut b: usize) -> Tensor {
        if a == b {
            return self.clone();
        }
       
        if b < a {  // Ensure a is always the smaller index
            std::mem::swap(&mut a, &mut b);
        }
        
        match self {
            Tensor::Scalar(v) => Tensor::Scalar(*v),
            Tensor::NTensor(ten, n_dim) => {
                if a == 0 {
                    if b == n_dim - 1 {  // Swap with innermost
                        let mut new = Vec::new();
                        for i in 0..ten[0].get_size(b - 1) {
                            new.push(Tensor::NTensor(
                                ten.iter().map(|t| t.get_element(b - 1, i)).collect(),
                                *n_dim
                            ));
                        }
                        Tensor::NTensor(new, *n_dim)
                    } else {  // Swap with other (not innermost)
                        let mut new = Vec::new();
                        for i in 0..ten[0].get_size(b - 1) {
                            new.push(Tensor::NTensor(
                                ten.iter().map(|t| t.get_slice(b - 1, i)).collect(),
                                *n_dim
                            ));
                        }
                        Tensor::NTensor(new, *n_dim)
                    }
                } else {  // Swap is 'deeper'
                    Tensor::NTensor(
                        ten.iter()
                           .map(|t| t.transpose_swap(a - 1, b - 1))
                           .collect(),
                        *n_dim
                    )
                }
            }
        }
    }

    fn transpose(&self, new_order: Vec<usize>) -> Tensor {
        match self {
            Tensor::Scalar(_) => self.clone(),
            Tensor::NTensor(ten, n_dim) => if new_order.len() != *n_dim {
                panic!("Transpose attempt had wrong amount of columns specified ({}), need {}", new_order.join(" "), &n_dim)
            } else if new_order.iter().any(|x| x >= &n_dim) {
                panic!("Transpose attempt had invalid index ({}), max is {} (0-based indexing)", new_order.join(" "), n_dim-1)
            } else {
                let mut new_t = self.clone();

                for (a, b) in get_least_swaps([0..n_dim-1].to_vec(), new_order.clone()) {
                    new_t = new_t.transpose_swap(a, b);
                }

                new_t
            }
        }
    }
}

fn get_least_swaps(old: &mut Vec<usize>, new: &mut Vec<usize>) -> Vec<(usize, usize)> {
    let n = old.len();
    let mut new_map: HashMap<usize, usize> = HashMap::new();
    let mut visited = vec![false; n];
    let mut swaps = Vec::new();

    // Create a map of element to index for the new vector
    for (i, &val) in new.iter().enumerate() {
        new_map.insert(val, i);
    }

    for i in 0..n {
        if visited[i] || old[i] == new[i] {
            continue;
        }

        // Start of a new cycle
        let mut j = i;
        while !visited[j] {
            visited[j] = true;
            let next = new_map[&old[j]];
            if next != j {
                swaps.push((j, next));
                old.swap(j, next);
            }
            j = next;
        }
    }

    swaps
}


// This iter shit is all ChatGPT, not gonna lie.
struct TensorIter<'a> {
    stack: Vec<&'a Tensor>,  // Stack to hold nested items for depth-first iteration
}

impl<'a> Iterator for TensorIter<'a> {
    type Item = &'a Tensor;

    fn next(&mut self) -> Option<Self::Item> {
        // Pop from the stack
        if let Some(current) = self.stack.pop() {
            match current {
                Tensor::Scalar(_) => Some(current),  // Return the scalar
                Tensor::NTensor(vec, _) => {
                    // Push the elements of the nested vector in reverse order onto the stack
                    for item in vec.iter().rev() {
                        self.stack.push(item);
                    }
                    self.next()  // Recursively call next() to get the next scalar or nested item
                }
            }
        } else {
            None  // No more items to iterate over
        }
    }
}

/* impl Tensor {
    fn _perform_matrix_operand<F>(&self, other: &Matrix, operand: F) -> Matrix  // Used for add,
                                                                                 // times, etc.
    where
        F: Fn(&f32, &f32) -> f32, 
    {
        Matrix::_create(&self.data
                             .iter()
                             .zip(other.data.iter())
                             .map(|(s_row, o_row)| {
                                 s_row.iter()
                                      .zip(o_row.iter())
                                      .map(|(s, o)| operand(s, o))
                                      .collect::<Vec<f32>>()
                             })
                             .collect::<Vec<Vec<f32>>>()
                            )
    }

    fn _perform_scalar_operand<F>(&self, scalar: &f32, operand: F) -> Matrix  // Used for add,
                                                                                 // times, etc.
    where
        F: Fn(&f32, &f32) -> f32, 
    {
        Matrix::_create(&self.data
                             .iter()
                             .map(|row| {
                                 row.iter()
                                    .map(|x| operand(x, scalar))
                                    .collect::<Vec<f32>>()
                             })
                             .collect::<Vec<Vec<f32>>>()
                            )
    }
} */


/* #[derive(Debug)]
struct Matrix {
    tensor: Tensor,
    shape: Vec<usize>,  
}

impl Matrix {
    fn _create(&self, t: &Tensor) -> Matrix {
        Matrix {
            tensor: t.clone(),
            shape: t.shape.clone()
        }
    }

    fn get_cofactor(&self, i: usize, j: usize) -> f32 {
        let minor_data: Vec<Vec<f32>> = self.tensor
                                            .data[0..i]
                                            .iter()
                                            .chain(self.data[i+1..].iter())
                                            .map(|row| row[0..j].iter()
                                                                .chain(row[j+1..].iter())
                                                                .cloned()
                                                                .collect()
                                                )
                                            .collect();
        let minor = Matrix::_create(&minor_data);

        f32::powi(-1.0, (i+j) as i32) * minor._calc_determinant()
    }

    fn get_cofactor_matrix(&self) -> Matrix {  // TODO: All this shit as attributes.
        Matrix::_create(&(0..self.shape[0]).map(|i| (0..self.shape[1]).map(|j| self.get_cofactor(i, j))
                                                                      .collect())
                                           .collect())
    }

    fn _calc_determinant(&self) -> f32 {  // TODO: Rewrite without recursion + matrix creation and
                                          // compare speed.
        if self.shape[0] == 2 {
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0];
        }

        (0..self.shape[1]).map(|j| self.get_cofactor(0, j) * self.data[0][j])
                          .sum()
    }

    fn get_determinant(&self) -> Result<f32, String> {
        if self.shape[0] != self.shape[1] {
            return Err(format!("Only square matrices have determinants ({}x{})", self.shape[0], self.shape[1]));
        }

        match self.shape[0] {
            0 => Ok(1.0),
            1 => Ok(self.data[0][0]),
            _ => Ok(self._calc_determinant())
        }
    }

    fn inverse(&self) -> Result<Matrix, String> {
        let det: f32;

        match self.get_determinant() {
            Ok(d) => if d == 0.0 {return Err("Det == 0, so no inverse matrix exists.".to_string())} else {det = d},
            Err(e) => return Err(e)
        }
        
        let adjugate = self.get_cofactor_matrix().transpose();

        Ok(&adjugate / det)
    }

    fn print(&self, title: &str) {
        println!("-------- {} ({}x{}) --------", title, self.shape[0], self.shape[1]);

        for row in &self.data {
            println!("{:?}", row);
        }

        println!("-----------------------------");
    }
} */

/* #[derive(Debug)]
struct Vector {
    tensor: Tensor,
    shape: Vec<usize>,  
} */


/* fn main() {
    let matrix1 = Matrix::from(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
    ])
    .expect("");

    matrix1.print("Base 1");

    let matrix2 = Matrix::from(vec![
        vec![1.0, 2.0, 3.0, 9.0],
        vec![4.0, 5.0, 6.0, 12.0],
        vec![3.0, 4.0, 8.0, 7.0],
        vec![2.0, 1.0, 1.0, 1.0],
    ])
    .expect("");

    matrix2.print("Base 2");

    (&matrix1 + &matrix1).expect(""). print("Double");
    (&matrix1 - 3.0).print("Plus 3");

    matrix1.transpose().print("Transposed");

    Matrix::identity(3).transpose().print("Identity");
    Matrix::filled(8.0, 3, 6).transpose().print("Filled");

    (&matrix1 * &matrix1.transpose()).expect("").print("Times");

    println!("DETERMINANT: {}", matrix2.get_determinant().expect(""));
    matrix2.inverse().expect("").print("Inverse");
} */

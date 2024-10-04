use std::collections::HashMap;
use std::iter::{FromIterator, Iterator, Sum};
use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};
use std::slice::Iter as SliceIter;


#[derive(Clone, Debug)]
pub enum Tensor {
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
            Tensor::NTensor(_, rank) => Tensor::NTensor(vec, rank+1)
        }
    }
}


// Iter shit was all Claude.ai
enum TensorIter<'a> {
    Scalar(std::iter::Once<&'a Tensor>),
    NTensor(SliceIter<'a, Tensor>),
}

impl<'a> Iterator for TensorIter<'a> {
    type Item = &'a Tensor;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            TensorIter::Scalar(iter) => iter.next(),
            TensorIter::NTensor(iter) => iter.next(),
        }
    }
}

// Implement IntoIterator for &Tensor to allow for..in loops
impl<'a> IntoIterator for &'a Tensor {
    type Item = &'a Tensor;
    type IntoIter = TensorIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}


impl Sum for Tensor {
    fn sum<I>(iter: I) -> Tensor
    where
        I: Iterator<Item = Tensor>,
    {
        iter.fold(Tensor::Scalar(0.0), |acc, x| &acc + &x)

    }
}


impl Tensor {
    pub fn iter(&self) -> TensorIter {
        match self {
            Tensor::Scalar(_) => TensorIter::Scalar(std::iter::once(self)),
            Tensor::NTensor(ref tensors, _) => TensorIter::NTensor(tensors.iter()),
        }
    }

    pub fn shape(&self) -> Vec<usize> {
        match self {
            Tensor::Scalar(_) => vec![],
            Tensor::NTensor(vec, _) => {
                let mut s = vec![vec.len()];
                s.extend(vec[0].shape());

                s
            }
        }
    }

    pub fn get_size(&self, dim: usize) -> usize {
        match self {
            Tensor::Scalar(_) => if dim == 0 {1} else {panic!("Scalars don't have dimensions")},
            Tensor::NTensor(ten, rank) => {
                if dim >= *rank {
                    panic!("Dimension {} is out of bounds for tensor of rank {}", dim, rank);
                }
                if dim == 0 {
                    ten.len()
                } else {
                    // Assume all sub-tensors have the same shape
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
            Tensor::NTensor(ten, rank) => {
                if dim == 0 {
                    ten[index].clone()
                } else {
                    Tensor::NTensor(
                        ten.iter().map(|t| t.get_slice(dim - 1, index)).collect(),
                        *rank
                    )
                }
            }
        }
    }
    
    pub fn rank(&self) -> usize {
        match self {
            Tensor::Scalar(_) => 0,
            Tensor::NTensor(_, rank) => *rank
        }
    }

    pub fn identity(size: usize, dimensions: usize) -> Tensor {
        if size <= 0 {
            panic!("Size must be > 0");
        }

        if dimensions == 0 {
            return Tensor::Scalar(1.0);
        }


        fn recursive_traverse(td: &Tensor, i: usize) -> Tensor {
            match td.clone() {
                Tensor::NTensor(mut vec, rank) => {
                    match vec[i] {
                        Tensor::Scalar(_) => {
                            vec[i] = Tensor::Scalar(1.0);
                        },
                        Tensor::NTensor(_, _) => {
                            vec[i] = recursive_traverse(&vec[i], i);
                        },
                    }
                    
                    Tensor::NTensor(vec, rank)
                },
                _ => panic!("Recursive traverse error, write this later!"),
            }
        }
        
        Tensor::filled(0.0, vec![size; dimensions]).iter()
                                                   .zip(0..size)
                                                   .map(|(t, i)| {
                                                        let mut t_clone = t.clone();
                                                        t_clone[i] = recursive_traverse(&t_clone[i], i);
                                                        t_clone
                                                   })
                                                   .collect()
    }

    pub fn filled(value: f32, dimensions: Vec<usize>) -> Tensor {
        if dimensions.is_empty() {
            return Tensor::Scalar(value);
        }

        Tensor::NTensor(vec![Tensor::filled(value, dimensions[1..].to_vec()); 
                             dimensions[0]], 
                        dimensions.len())
    }

    pub fn transpose(&self, mut a: usize, mut b: usize) -> Tensor {
        if a == b {
            return self.clone();
        }
       
        if b < a {  // Ensure a is always the smaller index
            std::mem::swap(&mut a, &mut b);
        }
        
        match self {
            Tensor::Scalar(v) => Tensor::Scalar(*v),
            Tensor::NTensor(ten, rank) => {
                if a == 0 {
                    if b == rank - 1 {  // Swap with innermost
                        let mut new = Vec::new();
                        for i in 0..ten[0].get_size(b - 1) {
                            new.push(Tensor::NTensor(
                                ten.iter().map(|t| t.get_element(b - 1, i)).collect(),
                                *rank
                            ));
                        }
                        Tensor::NTensor(new, *rank)
                    } else {  // Swap with other (not innermost)
                        let mut new = Vec::new();
                        for i in 0..ten[0].get_size(b - 1) {
                            new.push(Tensor::NTensor(
                                ten.iter().map(|t| t.get_slice(b - 1, i)).collect(),
                                *rank
                            ));
                        }
                        Tensor::NTensor(new, *rank)
                    }
                } else {  // Swap is 'deeper'
                    Tensor::NTensor(
                        ten.iter()
                           .map(|t| t.transpose(a - 1, b - 1))
                           .collect(),
                        *rank
                    )
                }
            }
        }
    }

    pub fn full_transpose(&self, new_order: Vec<usize>) -> Tensor {
        match self {
            Tensor::Scalar(_) => self.clone(),
            Tensor::NTensor(_, rank) => if new_order.len() != *rank {
                panic!("Transpose attempt had wrong amount of columns specified ({}), need {}", 
                    new_order.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" "),
                    &rank)
            } else if new_order.iter().any(|x| x >= &rank) {
                panic!("Transpose attempt had invalid index ({}), max is {} (0-based indexing)",
                    new_order.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" "),
                    rank-1)
            } else {
                let swaps = get_least_swaps(&mut (0..*rank as i32).map(|x| x as usize).collect(), &mut new_order.clone());
                swaps.iter().fold(self.clone(), |new, (a, b)| new.transpose(*a, *b))
            }
        }
    }

    pub fn tensor_product(&self, other: &Tensor) -> Tensor {
        match self {
            Tensor::Scalar(val) => other * *val,
            Tensor::NTensor(ten, rank) => Tensor::NTensor(ten.iter()
                                                             .map(|t| t.tensor_product(other))
                                                             .collect(),
                                                          rank + other.rank()),
        }
    }

    pub fn shape_as_string(&self) -> String {
       self.shape().iter().map(|x| x.to_string()).collect::<Vec<_>>().join("x") 
    }

    fn len(&self) -> usize {
        match self {
            Tensor::Scalar(_) => 1,
            Tensor::NTensor(data, _) => data.len(),
        }
    }

    pub fn contract(&self, ranks_to_contract: Vec<usize>) -> Tensor {
        match self {
            Tensor::Scalar(_) => panic!("Attempted to contract a scalar."),
            Tensor::NTensor(_, rank) => {
                if ranks_to_contract.is_empty() {
                    panic!("Contraction requires at least 1 dimension, 0 given");
                }
                if ranks_to_contract.iter().any(|&r| r >= *rank) {
                    panic!("Given ranks {:?} contain a rank higher than that of the tensor ({})", ranks_to_contract, rank);
                }
                if ranks_to_contract.len() != ranks_to_contract.iter().collect::<std::collections::HashSet<_>>().len() {
                    panic!("Duplicate ranks in contraction list");
                }

                // Reverse the ranks to match the new logic
                let reversed_ranks: Vec<usize> = ranks_to_contract.iter().map(|&r| rank - 1 - r).collect();
                let mut sorted_ranks = reversed_ranks;
                sorted_ranks.sort_unstable();

                // Check if dimensions to be contracted have the same length
                let dim_sizes: Vec<usize> = sorted_ranks.iter().map(|&r| self.get_size(r)).collect();
                if dim_sizes.windows(2).any(|w| w[0] != w[1]) {
                    panic!("Dimensions to be contracted must have the same length. Sizes: {:?}", dim_sizes);
                }

                // Helper function to perform contraction
                fn contract_helper(tensor: &Tensor, ranks: &[usize]) -> Tensor {
                    match tensor {
                        Tensor::Scalar(_) => panic!("Unexpected scalar encountered during contraction"),
                        Tensor::NTensor(data, rank) => {
                            if *rank < ranks.len() {
                                panic!("Cannot contract tensor with rank less than number of contraction dimensions");
                            }
                            if ranks.is_empty() {
                                return tensor.clone();
                            }
                            if ranks[0] == 0 {
                                // Base case: contract the first dimension
                                let result = data.iter().fold(Tensor::Scalar(0.0), |acc, t| &acc + t);
                                if ranks.len() == 1 {
                                    // If no more dimensions to contract, return the result
                                    result
                                } else {
                                    // Continue contracting remaining dimensions
                                    contract_helper(&result, &ranks[1..])
                                }
                            } else {
                                // Recursive case: move towards the dimensions to contract
                                Tensor::NTensor(
                                    data.iter()
                                        .map(|t| contract_helper(t, &ranks.iter().map(|&r| r - 1).collect::<Vec<_>>()))
                                        .collect(),
                                    rank - 1
                                )
                            }
                        }
                    }
                }

                contract_helper(self, &sorted_ranks)
            }
        }
    }

    fn perform_operand<F>(&self, other: &Tensor, operand: F) -> Tensor  // Used for add,
                                                                         // times, etc.
    where
        F: Fn(&f32, &f32) -> f32, 
    {
        // Neither scalar and unmatching.
        if self.shape() != other.shape() && (self.rank() > 0 && other.rank() > 0) {
            panic!("Mismatched dimensions: S ({}) - O ({})",
                   self.shape_as_string(), other.shape_as_string());
        }

        match (self, other) {
            (Tensor::Scalar(a), Tensor::Scalar(b)) => Tensor::Scalar(operand(a, b)),
            (Tensor::NTensor(ten_a, rank), Tensor::NTensor(ten_b, _)) => 
                if self.shape() != other.shape() {
                    panic!("Attemped operand on tensors of different shapes:\n\t[{}]\n\t[{}]",
                           self.shape_as_string(),
                           other.shape_as_string())
                } else {
                    Tensor::NTensor(ten_a.iter()
                                         .zip(ten_b)
                                         .map(|(a, b)| a.perform_operand(b, &operand))
                                         .collect::<Vec<_>>(),
                                    *rank)
                }
            (Tensor::NTensor(ten, rank), scalar) | (scalar, Tensor::NTensor(ten, rank)) => 
                Tensor::NTensor(ten.iter()
                                     .map(|t| t.perform_operand(scalar, &operand))
                                     .collect::<Vec<_>>(),
                                *rank)
        }
    }

    pub fn dot_product(&self, other: &Tensor) -> Tensor {
        match (self, other) {
            (Tensor::Scalar(a), Tensor::Scalar(b)) => Tensor::Scalar(a * b),
            (Tensor::NTensor(ten, rank), Tensor::Scalar(_)) | (Tensor::Scalar(_), Tensor::NTensor(ten, rank)) => 
                Tensor::NTensor(ten.iter()
                                   .map(|t| t.dot_product(other))
                                   .collect(), 
                                *rank),
            (Tensor::NTensor(_, _), Tensor::NTensor(_, rank)) => 
                self.tensor_product(other)  // Contracting innermost dim of LHS and outermost of RHS
                    .contract(vec![rank-1, *rank])
        }
    }

    pub fn hamarand_product(&self, other: &Tensor) -> Tensor {
        // Neither scalar and unmatching.
        if self.shape() != other.shape() && (self.rank() > 0 && other.rank() > 0) {
            panic!("Mismatched dimensions: S ({}) - O ({})",
                   self.shape_as_string(), other.shape_as_string());
        }

        Tensor::perform_operand(self, other, |x, y| Mul::mul(x, y))
    }
}


// All operators with 2 tensors are Hamarand style
// (except for multiplication, which is the dot product)

impl Add<&Tensor> for &Tensor {
    type Output = Tensor;

    fn add(self, other: &Tensor) -> Tensor {
        Tensor::perform_operand(self, other, |x, y| Add::add(x, y))
    }
}

impl Add<f32> for &Tensor {
    type Output = Tensor;

    fn add(self, other: f32) -> Tensor {
        self + &Tensor::Scalar(other)
    }
}

impl Add<f32> for Tensor {
    type Output = Tensor;

    fn add(self, other: f32) -> Tensor {
        &self + &Tensor::Scalar(other)
    }
}


impl Sub<&Tensor> for &Tensor {
    type Output = Tensor;

    fn sub(self, other: &Tensor) -> Tensor {
        Tensor::perform_operand(self, other, |x, y| Sub::sub(x, y))
    }
}

impl Sub<f32> for &Tensor {
    type Output = Tensor;

    fn sub(self, other: f32) -> Tensor {
        self + &Tensor::Scalar(other)
    }
}

impl Sub<f32> for Tensor {
    type Output = Tensor;

    fn sub(self, other: f32) -> Tensor {
        &self + &Tensor::Scalar(other)
    }
}


impl Div<&Tensor> for &Tensor {
    type Output = Tensor;

    fn div(self, other: &Tensor) -> Tensor {
        Tensor::perform_operand(self, other, |x, y| Div::div(x, y))
    }
}

impl Div<f32> for &Tensor {
    type Output = Tensor;

    fn div(self, other: f32) -> Tensor {
        self + &Tensor::Scalar(other)
    }
}

impl Div<f32> for Tensor {
    type Output = Tensor;

    fn div(self, other: f32) -> Tensor {
        &self + &Tensor::Scalar(other)
    }
}


impl Mul<&Tensor> for &Tensor {  // The fun one
    type Output = Tensor;

    fn mul(self, other: &Tensor) -> Tensor {
        self.dot_product(other)
    }
}

impl Mul<f32> for &Tensor {
    type Output = Tensor;

    fn mul(self, other: f32) -> Tensor {
        self + &Tensor::Scalar(other)
    }
}

impl Mul<f32> for Tensor {
    type Output = Tensor;

    fn mul(self, other: f32) -> Tensor {
        &self + &Tensor::Scalar(other)
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


// TODO
// Flatten(to dim_n)
// Normalize
    // sum(all): x^2 = 1
// Eigen
// Cross product (Vector)


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

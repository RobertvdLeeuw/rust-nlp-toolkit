use std::collections::HashMap;
use std::iter::{FromIterator, Iterator, Sum};
use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

/// The object that holds all the data for tensors. Its recursion allows tensors of arbitrary ranks
#[derive(Clone, Debug)]
pub enum TensorData {
    Scalar(f32),
    NTensor(Vec<TensorData>),
}

/// Implementation for common tools like indexing, creating from vec, and summing.
impl Index<usize> for TensorData {
    type Output = TensorData;
    fn index(&self, index: usize) -> &Self::Output {
        match self {
            TensorData::Scalar(_) => panic!("Attempted to index a scalar value"),
            TensorData::NTensor(vec) => &vec[index],
        }
    }
}

impl IndexMut<usize> for TensorData {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match self {
            TensorData::Scalar(_) => panic!("Attempted to index a scalar value"),
            TensorData::NTensor(vec) => &mut vec[index],
        }
    }
}

impl FromIterator<TensorData> for TensorData {
    fn from_iter<I: IntoIterator<Item = TensorData>>(iter: I) -> Self {
        TensorData::NTensor(iter.into_iter().collect())
    }
}

pub enum TensorDataIter<'a> {
    Scalar(std::iter::Once<&'a TensorData>),
    NTensor(std::slice::Iter<'a, TensorData>),
}

impl<'a> Iterator for TensorDataIter<'a> {
    type Item = TensorData;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            TensorDataIter::Scalar(iter) => iter.next().cloned(),
            TensorDataIter::NTensor(iter) => iter.next().cloned(),
        }
    }
}

impl Sum for TensorData {
    fn sum<I>(iter: I) -> TensorData
    where
        I: Iterator<Item = TensorData>,
    {
        iter.fold(TensorData::Scalar(0.0), |acc, x| {
            match (&acc, &x) {
                (TensorData::Scalar(a), TensorData::Scalar(b)) => TensorData::Scalar(a + b),
                _ => panic!("Cannot sum tensors of different ranks"),
            }
        })
    }
}


impl TensorData {
    /// Returns the size of the tensor. Example: 2x3 matrix -> [2, 3], vector of length 4 -> [4].
    pub fn shape(&self) -> Vec<usize> {
        match self {
            TensorData::Scalar(_) => vec![],
            TensorData::NTensor(vec) => {
                let mut s = vec![vec.len()];
                s.extend(vec[0].shape());

                s
            }
        }
    }

    pub fn iter(&self) -> TensorDataIter {
        match self {
            TensorData::Scalar(_) => TensorDataIter::Scalar(std::iter::once(self)),
            TensorData::NTensor(vec) => TensorDataIter::NTensor(vec.iter()),
        }
    }

    pub fn shape_as_string(&self) -> String {
       self.shape().iter().map(|x| x.to_string()).collect::<Vec<_>>().join("x") 
    }

    pub fn filled(value: f32, shape: Vec<usize>) -> TensorData {
        if shape.is_empty() {  // Reached 'bottom', base case.'
            return TensorData::Scalar(value);
        }

        TensorData::NTensor(vec![TensorData::filled(value, shape[1..].to_vec()); 
                                 shape[0]])
    }

    /// Used for add, times, etc.
    pub fn perform_operand<F>(&self, other: &TensorData, operand: F) -> TensorData
    where
        F: Fn(f32, f32) -> f32 + Copy, 
    {
        if self.shape() != other.shape() && (self.rank() > 0 && other.rank() > 0){
            panic!("Mismatched shapes: S ({}) - O ({})",
                   self.shape_as_string(), other.shape_as_string());
        }

        fn apply_operand<F>(a: &TensorData, b: &TensorData, operand: F) -> TensorData
        where
            F: Fn(f32, f32) -> f32 + Copy,
        {
            match (a, b) {
                (TensorData::Scalar(x), TensorData::Scalar(y)) => TensorData::Scalar(operand(*x, *y)),
                (TensorData::NTensor(ten_a), TensorData::NTensor(ten_b)) => {
                    TensorData::NTensor(
                        ten_a.iter()
                            .zip(ten_b.iter())
                            .map(|(a, b)| apply_operand(a, b, operand))
                            .collect()
                    )
                },
                (TensorData::NTensor(ten), TensorData::Scalar(scalar)) |
                (TensorData::Scalar(scalar), TensorData::NTensor(ten)) => {
                    TensorData::NTensor(
                        ten.iter()
                            .map(|t| apply_operand(t, &TensorData::Scalar(*scalar), operand))
                            .collect()
                    )
                },
                _ => unreachable!("Shapes were checked at the beginning of perform_operand"),
            }
        }

        apply_operand(self, other, operand)
    }

    pub fn len(&self, dim: usize) -> usize {
        match self {
            TensorData::Scalar(_) => if dim == 0 {1} else {panic!("Scalars don't have dimensions")},
            TensorData::NTensor(ten) => {
                if dim >= self.rank() {
                    panic!("Dimension {} is out of bounds for tensor of rank {}", dim, self.rank());
                }
                if dim == 0 {
                    ten.len()
                } else {
                    // Assume all sub-tensors have the same shape
                    ten[0].len(dim - 1)
                }
            }
        }
    }

    pub fn rank(&self) -> usize {
        match self {
            TensorData::Scalar(_) => 0,
            TensorData::NTensor(vec) => vec[0].rank() + 1,
        }
    }

    /// Basically replaces every scalar in self with other scaled by it, and returns a tensor(data)
    /// of rank (self.rank + other.rank)
    pub fn tensor_product(&self, other: &TensorData) -> TensorData {
        match self {
            TensorData::Scalar(val) => other * *val,
            TensorData::NTensor(ten) => TensorData::NTensor(ten.iter()
                                                               .map(|t| t.tensor_product(other))
                                                               .collect())
        }
    }

    /// Shrinks the tensor (by the emount of contracted ranks). I do not know how to explain this
    /// in brief, watch a video about it.
    pub fn contract(&self, ranks_to_contract: Vec<usize>) -> TensorData {
        match self {
            TensorData::Scalar(_) => panic!("Attempted to contract a scalar."),
            TensorData::NTensor(_) => {
                if ranks_to_contract.is_empty() {
                    panic!("Contraction requires at least 1 dimension, 0 given");
                }
                if ranks_to_contract.iter().any(|&r| r >= self.rank()) {
                    panic!("Given ranks {:?} contain a rank higher than that of the tensor ({})", ranks_to_contract, self.rank());
                }
                if ranks_to_contract.len() != ranks_to_contract.iter().collect::<std::collections::HashSet<_>>().len() {
                    panic!("Duplicate ranks in contraction list");
                }

                // Reverse the ranks to match the new logic
                let reversed_ranks: Vec<usize> = ranks_to_contract.iter().map(|&r| self.rank() - 1 - r).collect();
                let mut sorted_ranks = reversed_ranks;
                sorted_ranks.sort_unstable();

                // Check if dimensions to be contracted have the same length
                let dim_sizes: Vec<usize> = sorted_ranks.iter().map(|&r| self.len(r)).collect();
                if dim_sizes.windows(2).any(|w| w[0] != w[1]) {
                    panic!("Dimensions to be contracted must have the same length. Sizes: {:?}", dim_sizes);
                }

                // Helper function to perform contraction
                fn contract_helper(tensor: &TensorData, ranks: &[usize]) -> TensorData {
                    match tensor {
                        TensorData::Scalar(_) => panic!("Unexpected scalar encountered during contraction"),
                        TensorData::NTensor(data) => {
                            if tensor.rank() < ranks.len() {
                                panic!("Cannot contract tensor with rank less than number of contraction dimensions");
                            }
                            if ranks.is_empty() {
                                return tensor.clone();
                            }
                            if ranks[0] == 0 {
                                // Base case: contract the first dimension
                                let result = data.iter().fold(TensorData::Scalar(0.0), |acc, t| &acc + t);
                                if ranks.len() == 1 {
                                    // If no more dimensions to contract, return the result
                                    result
                                } else {
                                    // Continue contracting remaining dimensions
                                    contract_helper(&result, &ranks[1..])
                                }
                            } else {
                                // Recursive case: move towards the dimensions to contract
                                TensorData::NTensor(
                                    data.iter()
                                        .map(|t| contract_helper(t, &ranks.iter().map(|&r| r - 1).collect::<Vec<_>>()))
                                        .collect())
                            }
                        }
                    }
                }

                contract_helper(self, &sorted_ranks)
            }
        }
    }

    /// Returns a tensor of the rank given rank and size
    /// where T[i][j]..[n] = 1 if i==j==...==n else 0.
    /// Basically filled with 0 but with a line of 1's diagonal through all dimensions.
    pub fn identity(size: usize, rank: usize) -> TensorData {
        if size <= 0 {
            panic!("Size must be > 0");
        }

        if rank == 0 {
            return TensorData::Scalar(1.0);
        }


        fn recursive_traverse(td: &TensorData, i: usize) -> TensorData {
            match td.clone() {
                TensorData::NTensor(mut vec) => {
                    match vec[i] {
                        TensorData::Scalar(_) => {
                            vec[i] = TensorData::Scalar(1.0);
                        },
                        TensorData::NTensor(_) => {
                            vec[i] = recursive_traverse(&vec[i], i);
                        },
                    }
                    
                    TensorData::NTensor(vec)
                },
                TensorData::Scalar(_) => {
                    TensorData::Scalar(1.0)
                }
            }
        }
        
        TensorData::filled(0.0, vec![size; rank]).iter()
                                                 .zip(0..size)
                                                 .map(|(t, i)| {
                                                      let mut t_clone = t.clone();
                                                      t_clone[i] = recursive_traverse(&t_clone[i], i);
                                                      t_clone
                                                 })
                                                       .collect()
    }

    /// Helper function
    pub fn get_element(&self, dim: usize, index: usize) -> TensorData {
        match self {
            TensorData::Scalar(_) => self.clone(),
            TensorData::NTensor(ten) => {
                if dim == 0 {
                    ten[index].clone()
                } else {
                    ten[0].get_element(dim - 1, index)
                }
            }
        }
    }

    /// Helper function
    pub fn get_slice(&self, dim: usize, index: usize) -> TensorData {
        match self {
            TensorData::Scalar(_) => self.clone(),
            TensorData::NTensor(ten) => {
                if dim == 0 {
                    ten[index].clone()
                } else {
                    TensorData::NTensor(
                        ten.iter().map(|t| t.get_slice(dim - 1, index)).collect()
                    )
                }
            }
        }
    }

   pub fn transpose(&self, mut a: usize, mut b: usize) -> TensorData {
        if a == b {
            return self.clone();
        }
       
        if b < a {  // Ensure a is always the smaller index
            std::mem::swap(&mut a, &mut b);
        }
        
        match self {
            TensorData::Scalar(v) => TensorData::Scalar(*v),
            TensorData::NTensor(ten) => {
                if a == 0 {
                    if b == self.rank() - 1 {  // Swap with innermost
                        let mut new = Vec::new();
                        for i in 0..ten[0].len(b - 1) {
                            new.push(TensorData::NTensor(
                                ten.iter().map(|t| t.get_element(b - 1, i)).collect()
                            ));
                        }
                        TensorData::NTensor(new)
                    } else {  // Swap with other (not innermost)
                        let mut new = Vec::new();
                        for i in 0..ten[0].len(b - 1) {
                            new.push(TensorData::NTensor(
                                ten.iter().map(|t| t.get_slice(b - 1, i)).collect()
                            ));
                        }
                        TensorData::NTensor(new)
                    }
                } else {  // Swap is 'deeper'
                    TensorData::NTensor(
                        ten.iter()
                           .map(|t| t.transpose(a - 1, b - 1))
                           .collect()
                    )
                }
            }
        }
    }

    /// A collection of transposes requested via the entire reordering of the dimensions.
    pub fn full_transpose(&self, new_order: Vec<usize>) -> TensorData {
        match self {
            TensorData::Scalar(_) => self.clone(),
            TensorData::NTensor(_) => if new_order.len() != self.rank() {
                panic!("Transpose attemUpdate main.rspt had wrong amount of columns specified ({}), need {}", 
                    new_order.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" "),
                    &self.rank())
            } else if new_order.iter().any(|x| x >= &self.rank()) {
                panic!("Transpose attempt had invalid index ({}), max is {} (0-based indexing)",
                    new_order.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" "),
                    self.rank()-1)
            } else {
                let swaps = get_least_swaps(&mut (0..self.rank() as i32).map(|x| x as usize).collect(), &mut new_order.clone());
                swaps.iter().fold(self.clone(), |new, (a, b)| new.transpose(*a, *b))
            }
        }
    }

    /// Typical matrix multiplication generalized to all tensors. 
    /// Returns the tensor product of self and other contranked on the innermost dimension of self
    /// and outermost of other.
    pub fn dot_product(&self, other: &TensorData) -> TensorData {
        match (self, other) {
            (TensorData::Scalar(a), TensorData::Scalar(b)) => TensorData::Scalar(a * b),
            (TensorData::NTensor(ten), TensorData::Scalar(_)) | (TensorData::Scalar(_), TensorData::NTensor(ten)) => 
                TensorData::NTensor(ten.iter()
                                   .map(|t| t.dot_product(other))
                                   .collect()),
            (TensorData::NTensor(_), TensorData::NTensor(_)) => 
                self.tensor_product(other)  // Contracting innermost dim of LHS and outermost of RHS
                    .contract(vec![self.rank()-1, self.rank()])
        }
    }

    /// T1 * T2, where a new tensor of the same shape is returned 
    /// where T[i][j]... = T1[i][j]... * T2[i][j]...
    pub fn hamarand_product(&self, other: &TensorData) -> TensorData {
        // Neither scalar and unmatching.
        if self.shape() != other.shape() && (self.rank() > 0 && other.rank() > 0) {
            panic!("Mismatched dimensions: S ({}) - O ({})",
                   self.shape_as_string(), other.shape_as_string());
        }

        TensorData::perform_operand(self, other, Mul::mul)
    }

    pub fn value(&self) -> f32 {
        match self {
            TensorData::Scalar(val) => *val,
            TensorData::NTensor(_) => panic!("Attempted to retrieve the value of a non-scalar.")
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


// All operators with 2 tensors are Hamarand style
// (except for multiplication, which is the dot product).

macro_rules! impl_tensor_data_op {
    ($trait:ident, $method:ident, $op:expr) => {
        impl $trait for &TensorData {
            type Output = TensorData;
            fn $method(self, rhs: Self) -> Self::Output {
                self.perform_operand(rhs, $op)
            }
        }

        impl $trait<f32> for &TensorData {
            type Output = TensorData;
            fn $method(self, rhs: f32) -> Self::Output {
                self.perform_operand(&TensorData::Scalar(rhs), $op)
            }
        }

        impl $trait<f32> for TensorData {
            type Output = TensorData;
            fn $method(self, rhs: f32) -> Self::Output {
                self.perform_operand(&TensorData::Scalar(rhs), $op)
            }
        }

        impl $trait<&TensorData> for f32 {
            type Output = TensorData;
            fn $method(self, rhs: &TensorData) -> Self::Output {
                rhs.perform_operand(&TensorData::Scalar(self), |a, b| $op(b, a))
            }
        }

        impl $trait<TensorData> for f32 {
            type Output = TensorData;
            fn $method(self, rhs: TensorData) -> Self::Output {
                (&rhs).perform_operand(&TensorData::Scalar(self), |a, b| $op(b, a))
            }
        }
    }
}

impl_tensor_data_op!(Add, add, |a, b| a + b);
impl_tensor_data_op!(Sub, sub, |a, b| a - b);
impl_tensor_data_op!(Div, div, |a, b| a / b);

// Special implementation for Mul to handle dot product for TensorData
impl Mul for &TensorData {
    type Output = TensorData;
    fn mul(self, rhs: Self) -> Self::Output {
        self.dot_product(rhs)
    }
}

impl Mul<f32> for &TensorData {
    type Output = TensorData;
    fn mul(self, rhs: f32) -> Self::Output {
        self.perform_operand(&TensorData::Scalar(rhs), |a, b| a * b)
    }
}

impl Mul<f32> for TensorData {
    type Output = TensorData;
    fn mul(self, rhs: f32) -> Self::Output {
        self.perform_operand(&TensorData::Scalar(rhs), |a, b| a * b)
    }
}

impl Mul<&TensorData> for f32 {
    type Output = TensorData;
    fn mul(self, rhs: &TensorData) -> Self::Output {
        rhs.perform_operand(&TensorData::Scalar(self), |a, b| a * b)
    }
}

impl Mul<TensorData> for f32 {
    type Output = TensorData;
    fn mul(self, rhs: TensorData) -> Self::Output {
        (&rhs).perform_operand(&TensorData::Scalar(self), |a, b| a * b)
    }
}



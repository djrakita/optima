
/// # Example:
/// permutations_of_ranges(vec!\[3,2]) should return the following:
/// \[\[0,0], \[0,1], \[1,0], \[1,1], \[2, 0], \[2, 1]]
pub fn permutations_of_ranges(ranges: Vec<usize>) -> Vec<Vec<usize>> {
    let mut out_vec = vec![];

    let mut initial_vec = vec![];
    for _ in &ranges { initial_vec.push(0); }

    out_vec.push(initial_vec);

    for (idx, r) in ranges.iter().enumerate() {
        for i in 0..*r {
            if i == 0 { continue; }
            let mut tmp = vec![];
            for v in out_vec.iter() {
                let mut v_clone = v.clone();
                v_clone[idx] = i;
                tmp.push(v_clone);
            }

            for t in tmp { out_vec.push(t); }
        }
    }


    out_vec
}
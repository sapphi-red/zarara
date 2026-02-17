#![cfg_attr(not(test), allow(dead_code))]

pub(crate) fn has_cycle(node_count: usize, edges: &[(usize, usize)]) -> bool {
    let mut adjacency = vec![Vec::new(); node_count];
    for &(from, to) in edges {
        if from < node_count && to < node_count {
            adjacency[from].push(to);
        }
    }

    let mut visit_state = vec![0_u8; node_count];
    for node in 0..node_count {
        if visit_state[node] == 0 && dfs_has_cycle(node, &adjacency, &mut visit_state) {
            return true;
        }
    }

    false
}

pub(crate) fn find_cycle(node_count: usize, edges: &[(usize, usize)]) -> Option<Vec<usize>> {
    let mut adjacency = vec![Vec::new(); node_count];
    for &(from, to) in edges {
        if from < node_count && to < node_count {
            adjacency[from].push(to);
        }
    }

    let mut visit_state = vec![0_u8; node_count];
    let mut stack = Vec::new();
    let mut stack_pos = vec![None; node_count];
    for node in 0..node_count {
        if visit_state[node] == 0 {
            let cycle = dfs_find_cycle(
                node,
                &adjacency,
                &mut visit_state,
                &mut stack,
                &mut stack_pos,
            );
            if cycle.is_some() {
                return cycle;
            }
        }
    }

    None
}

fn dfs_has_cycle(node: usize, adjacency: &[Vec<usize>], visit_state: &mut [u8]) -> bool {
    visit_state[node] = 1;
    for &next in &adjacency[node] {
        if visit_state[next] == 1 {
            return true;
        }
        if visit_state[next] == 0 && dfs_has_cycle(next, adjacency, visit_state) {
            return true;
        }
    }
    visit_state[node] = 2;
    false
}

fn dfs_find_cycle(
    node: usize,
    adjacency: &[Vec<usize>],
    visit_state: &mut [u8],
    stack: &mut Vec<usize>,
    stack_pos: &mut [Option<usize>],
) -> Option<Vec<usize>> {
    visit_state[node] = 1;
    stack_pos[node] = Some(stack.len());
    stack.push(node);

    for &next in &adjacency[node] {
        if visit_state[next] == 1 {
            let start = stack_pos[next].unwrap_or(0);
            let mut cycle = stack[start..].to_vec();
            cycle.push(next);
            return Some(cycle);
        }
        if visit_state[next] == 0 {
            let cycle = dfs_find_cycle(next, adjacency, visit_state, stack, stack_pos);
            if cycle.is_some() {
                return cycle;
            }
        }
    }

    stack.pop();
    stack_pos[node] = None;
    visit_state[node] = 2;
    None
}

#[cfg(test)]
mod tests {
    use super::{find_cycle, has_cycle};

    #[test]
    fn dag_has_no_cycle() {
        let edges = vec![(0, 1), (0, 2), (2, 3)];
        assert!(!has_cycle(4, &edges));
    }

    #[test]
    fn known_cycle_is_detected() {
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        assert!(has_cycle(3, &edges));
    }

    #[test]
    fn finds_cycle_path() {
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        let cycle = find_cycle(3, &edges).expect("expected cycle path");
        assert!(cycle.len() >= 4);
        assert_eq!(cycle.first(), cycle.last());
    }

    #[test]
    fn no_cycle_returns_none() {
        let edges = vec![(0, 1), (1, 2)];
        assert!(find_cycle(3, &edges).is_none());
    }
}

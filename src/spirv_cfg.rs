/*
 * Copyright 2016-2019 Arm Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

use std::collections::{HashMap, HashSet};

use crate::spirv_common::{
    Merge,
    Terminator,
    SPIRFunction,
    VariantHolder,
};

use crate::spirv_cross::{
    Compiler,
};

struct VisitOrder {
    v: i32,
}

impl VisitOrder {
    fn get(&self) -> i32 {
        self.v
    }
}

impl Default for VisitOrder {
    fn default() -> Self {
        VisitOrder { v: -1 }
    }
}


struct Cfg<'a> {
    compiler: &'a Compiler,
    func: &'a SPIRFunction,
    preceding_edges: HashMap<u32, Vec<u32>>,
    succeeding_edges: HashMap<u32, Vec<u32>>,
    immediate_dominators: HashMap<u32, u32>,
    visit_order: HashMap<u32, VisitOrder>,
    post_order: Vec<u32>,
    empty_vector: Vec<u32>,
    visit_count: u32,
}

impl<'a> Cfg<'a> {
    fn new(compiler: &'a Compiler, func: &'a SPIRFunction) -> Self {
        let mut cfg = Cfg {
            compiler,
            func,
            preceding_edges: HashMap::new(),
            succeeding_edges: HashMap::new(),
            immediate_dominators: HashMap::new(),
            visit_order: HashMap::new(),
            post_order: Vec::new(),
            empty_vector: Vec::new(),
            visit_count: 0,
        };
        cfg.build_post_order_visit_order();
        cfg.build_immediate_dominators();
        cfg
    }

    fn get_compiler(&self) -> &'a Compiler {
        self.compiler
    }

    fn get_function(&self) -> &'a SPIRFunction {
        self.func
    }

    fn get_immediate_dominator(&self, block: u32) -> u32 {
        if let Some(value) = self.immediate_dominators.get(&block) {
            return *value;
        }
        0
    }

    fn get_visit_order(&self, block: u32) -> u32 {
        let value = self.visit_order
            .get(&block)
            .unwrap();
        let v = value.v;
        assert!(v > 0);
        return v as u32;
    }

    fn find_common_dominator(&self, mut a: u32, mut b: u32) -> u32 {
        while a != b {
            if self.get_visit_order(a) < self.get_visit_order(b) {
                a = self.get_immediate_dominator(a);
            } else {
                b = self.get_immediate_dominator(b);
            }
        }
        return a;
    }

    fn build_immediate_dominators(&mut self) {
        // Traverse the post-order in reverse and build up the immediate dominator tree.
        self.immediate_dominators.clear();
        self.immediate_dominators.insert(self.func.entry_block, self.func.entry_block);

        for block in self.post_order.iter().rev() {
            if let Some(pred) = &self.preceding_edges.get(block) {
                for edge in *pred {
                    if self.immediate_dominators.contains_key(block) {
                        assert!(self.immediate_dominators.contains_key(edge));
                        self.immediate_dominators.insert(*block, self.find_common_dominator(*block, *edge));
                    } else {
                        self.immediate_dominators.insert(*block, *edge);
                    }
                }
            }
        }
    }

    fn is_back_edge(&self, to: u32) -> bool {
        // We have a back edge if the visit order is set with the temporary magic value 0.
        // Crossing edges will have already been recorded with a visit order.
        if let Some(value) = self.visit_order.get(&to) {
            return value.v == 0;
        }
        panic!("is_back_edge - 'to' not found");
    }

    fn post_order_visit(&mut self, block_id: u32) -> bool {
        // If we have already branched to this block (back edge), stop recursion.
        // If our branches are back-edges, we do not record them.
        // We have to record crossing edges however.
        if let Some(mut order) = self.visit_order.get_mut(&block_id) {
            if order.v >= 0 {
                return !self.is_back_edge(block_id);
            }
            // Block back-edges from recursively revisiting ourselves.
            order.v = 0;
        } else {
            // uncovered branch in the original code
            return false;
        }

        // First visit our branch targets.
        let block = match self.compiler.get(block_id as usize) {
            VariantHolder::SPIRBlock(value) => value,
            _ => panic!("Bad cast"),
        };
        match block.terminator {
            Terminator::Direct => {
                if self.post_order_visit(block.next_block) {
                    self.add_branch(block_id, block.next_block);
                }
            },
            Terminator::Select => {
                if self.post_order_visit(block.true_block) {
                    self.add_branch(block_id, block.true_block);
                }
                if self.post_order_visit(block.false_block) {
                    self.add_branch(block_id, block.false_block);
                }
            },
            Terminator::MultiSelect => {
                for target in block.cases {
                    if self.post_order_visit(target.block) {
                        self.add_branch(block_id, target.block);
                    }
                }
                if block.default_block != 0 && self.post_order_visit(block.default_block) {
                    self.add_branch(block_id, block.default_block);
                }
            },
            _ => (),
        }

        // If this is a loop header, add an implied branch to the merge target.
        // This is needed to avoid annoying cases with do { ... } while(false) loops often generated by inliners.
        // To the CFG, this is linear control flow, but we risk picking the do/while scope as our dominating block.
        // This makes sure that if we are accessing a variable outside the do/while, we choose the loop header as dominator.
        if block.merge as u32 == Merge::MergeLoop as u32 {
            self.add_branch(block_id, block.merge_block);
        }

        // Then visit ourselves. Start counting at one, to let 0 be a magic value for testing back vs. crossing edges.
        self.visit_count += 1;
        if let Some(order) = self.visit_order.get_mut(&block_id) {
            order.v = self.visit_count as i32;
        };
        self.post_order.push(block_id);
        return true;
    }

    fn build_post_order_visit_order(&mut self) {
        let block = self.func.entry_block;
        self.visit_count = 0;
        self.visit_order.clear();
        self.post_order.clear();
        self.post_order_visit(block);
    }

    fn add_branch(&mut self, from: u32, to: u32) {
        let add_unique = |l: &mut Vec<u32>, value: u32| {
            if let None = l.iter().find(|x| **x == value) {
                l.push(value);
            }
        };
        if let Some(edge) = self.preceding_edges.get_mut(&to) {
            add_unique(edge, from);
        }
        if let Some(edge) = self.succeeding_edges.get_mut(&from) {
            add_unique(edge, to);
        }
    }

    fn get_preceding_edges(&'a self, block: u32) -> &'a Vec<u32> {
        if let Some(edge) = self.preceding_edges.get(&block) {
            return edge;
        }
        &vec![]
    }

    fn get_succeeding_edges(&'a self, block: u32) -> &'a Vec<u32> {
        if let Some(edge) = self.succeeding_edges.get(&block) {
            return edge;
        }
        &vec![]
    }

    fn walk_from(&self, seen_blocks: &mut HashSet<u32>, block: u32, op: impl Fn(u32) -> ()) {
        if seen_blocks.contains(&block) {
            return;
        }
        seen_blocks.insert(block);
        op(block);
        for b in self.get_succeeding_edges(block) {
            self.walk_from(seen_blocks, *b, op);
        }
    }

    // private
//    fn add_branch(&self, from: u32, to: u32) {
//
//    }
//    fn build_post_order_visit_order(&self) {
//
//    }
//    fn build_immediate_dominators(&self) {
//
//    }
//    fn post_order_visit(&self, block: u32) -> bool {
//
//    }

//    fn is_back_edge(&self, to: u32) -> bool {
//
//    }
}

struct DominatorBuilder<'a> {
    cfg: &'a Cfg<'a>,
    dominator: u32,
}

impl<'a> DominatorBuilder<'a> {
    fn new(cfg: &'a Cfg<'a>) -> Self {
        DominatorBuilder { cfg, dominator: 0 }
    }

    fn add_block(&mut self, block: u32) {
        if self.cfg.get_immediate_dominator(block) == 0 {
            // Unreachable block via the CFG, we will never emit this code anyways.
            return;
        }

        if self.dominator == 0 {
            self.dominator = block;
            return;
        }

        if block != self.dominator {
            self.dominator = self.cfg.find_common_dominator(block, self.dominator);
        }
    }
    fn get_dominator(&self) -> u32 {
        self.dominator
    }
    fn lift_continue_block_dominator(&mut self) {
        // It is possible for a continue block to be the dominator of a variable is only accessed inside the while block of a do-while loop.
        // We cannot safely declare variables inside a continue block, so move any variable declared
        // in a continue block to the entry block to simplify.
        // It makes very little sense for a continue block to ever be a dominator, so fall back to the simplest
        // solution.

        if self.dominator == 0 {
            return;
        }

        let block = match self.cfg.get_compiler().get(self.dominator as usize) {
            VariantHolder::SPIRBlock(value) => value,
            _ => panic!("Bad cast"),
        };
        let post_order = self.cfg.get_visit_order(self.dominator);

        // If we are branching to a block with a higher post-order traversal index (continue blocks), we have a problem
        // since we cannot create sensible GLSL code for this, fallback to entry block.
        let mut back_edge_dominator = false;
        match block.terminator {
            Terminator::Direct => {
                if self.cfg.get_visit_order(block.next_block) > post_order {
                    back_edge_dominator = true;
                }
            },
            Terminator::Select => {
                if self.cfg.get_visit_order(block.true_block) > post_order {
                    back_edge_dominator = true;
                }
                if self.cfg.get_visit_order(block.false_block) > post_order {
                    back_edge_dominator = true;
                }
            },
            Terminator::MultiSelect => {
                for target in block.cases {
                    if self.cfg.get_visit_order(target.block) > post_order {
                        back_edge_dominator = true;
                    }
                }
                if block.default_block != 0 && self.cfg.get_visit_order(block.default_block) > post_order {
                    back_edge_dominator = true;
                }
            },
            _ => (),
        }

        if back_edge_dominator {
            self.dominator = self.cfg.get_function().entry_block;
        }
    }
}

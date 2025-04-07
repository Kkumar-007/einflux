import unittest
import numpy as np
from io import StringIO
import sys
import re

# Import the module to test - assuming the code is in a file named rearrange.py
# If using a different file structure, adjust the import statement accordingly
from rearrange import rearrange, RearrangeError, parse_pattern, expand_dims, get_permutation, reshape_for_output


class TestParsePattern(unittest.TestCase):
    """Tests for the parse_pattern function"""
    
    def test_basic_pattern(self):
        """Test parsing of basic patterns"""
        self.assertEqual(parse_pattern("a b c"), ["a", "b", "c"])
        self.assertEqual(parse_pattern("a"), ["a"])
        self.assertEqual(parse_pattern("a b"), ["a", "b"])
        
    def test_tuple_pattern(self):
        """Test parsing of patterns with tuples"""
        self.assertEqual(parse_pattern("(a b)"), [("a", "b")])
        self.assertEqual(parse_pattern("a (b c)"), ["a", ("b", "c")])
        self.assertEqual(parse_pattern("(a b) c"), [("a", "b"), "c"])
        self.assertEqual(parse_pattern("(a b) (c d)"), [("a", "b"), ("c", "d")])
        
    def test_mixed_pattern(self):
        """Test parsing of mixed patterns with tuples and digits"""
        self.assertEqual(parse_pattern("a (b 2)"), ["a", ("b", "2")])
        self.assertEqual(parse_pattern("(a 3) b"), [("a", "3"), "b"])
        
    def test_wildcard_pattern(self):
        """Test parsing of patterns with wildcards"""
        self.assertEqual(parse_pattern("* a b"), ["*", "a", "b"])
        self.assertEqual(parse_pattern("a ... b"), ["a", "...", "b"])
        self.assertEqual(parse_pattern("a * (b c)"), ["a", "*", ("b", "c")])
        
    def test_complex_pattern(self):
        """Test parsing of complex patterns"""
        self.assertEqual(parse_pattern("a (b c) d * (e f)"), 
                        ["a", ("b", "c"), "d", "*", ("e", "f")])


class TestExpandDims(unittest.TestCase):
    """Tests for the expand_dims function"""
    
    def test_basic_expand(self):
        """Test basic dimension expansion"""
        shape = [2, 3, 4]
        structure = ["a", "b", "c"]
        shape_dict = {}
        result = expand_dims(shape, structure, shape_dict)
        
        self.assertEqual(result, [2, 3, 4])
        self.assertEqual(shape_dict, {"a": 2, "b": 3, "c": 4})
        
    def test_tuple_expand_split(self):
        """Test tuple expansion when splitting dimensions"""
        shape = [6, 4]
        structure = [("a", "b"), "c"]
        shape_dict = {"a": 2, "b": 3}
        result = expand_dims(shape, structure, shape_dict, merging=False)
        
        self.assertEqual(result, [2, 3, 4])
        self.assertEqual(shape_dict, {"a": 2, "b": 3, "c": 4})
        
    def test_tuple_expand_merge(self):
        """Test tuple expansion when merging dimensions"""
        shape = [2, 3, 4]
        structure = ["a", "b", "c"]
        shape_dict = {"a": 2, "b": 3, "c": 4}
        
        result = expand_dims([2, 3], [("a", "b")], shape_dict, merging=True)
        self.assertEqual(result, [6])
        
    def test_ellipsis_expand(self):
        """Test ellipsis dimension expansion"""
        shape = [2, 3, 4, 5, 6]
        structure = ["a", "...", "b"]
        shape_dict = {}
        result = expand_dims(shape, structure, shape_dict)
        
        self.assertEqual(result, [2, 3, 4, 5, 6])
        self.assertEqual(shape_dict, {"a": 2, "b": 6, "...": [3, 4, 5]})
        
    def test_error_dimensions(self):
        """Test dimension mismatch error"""
        shape = [2, 3]
        structure = ["a", "b", "c"]
        shape_dict = {}
        
        with self.assertRaises(RearrangeError):
            expand_dims(shape, structure, shape_dict)
            
    def test_error_product_mismatch(self):
        """Test product mismatch error"""
        shape = [5, 3]
        structure = [("a", "b"), "c"]
        shape_dict = {"a": 2, "b": 3}  # Product is 6, but shape has 5
        
        with self.assertRaises(RearrangeError):
            expand_dims(shape, structure, shape_dict, merging=False)


class TestGetPermutation(unittest.TestCase):
    """Tests for the get_permutation function"""
    
    def test_basic_permutation(self):
        """Test basic dimension permutation"""
        input_structure = ["a", "b", "c"]
        output_structure = ["c", "a", "b"]
        shape_dict = {"a": 2, "b": 3, "c": 4}
        
        result = get_permutation(input_structure, output_structure, shape_dict)
        self.assertEqual(result, [2, 0, 1])
        
    def test_tuple_permutation(self):
        """Test permutation with tuples"""
        input_structure = ["a", ("b", "c")]
        output_structure = [("c", "b"), "a"]
        shape_dict = {"a": 2, "b": 3, "c": 4}
        
        result = get_permutation(input_structure, output_structure, shape_dict)
        self.assertEqual(result, [2, 1, 0])
        
    def test_wildcard_permutation(self):
        """Test permutation with wildcards"""
        input_structure = ["a", "*", "b"]
        output_structure = ["b", "*", "a"]
        shape_dict = {"a": 2, "b": 5, "*": [3, 4]}
        
        result = get_permutation(input_structure, output_structure, shape_dict)
        self.assertEqual(result, [3, 1, 2, 0])


class TestReshapeForOutput(unittest.TestCase):
    """Tests for the reshape_for_output function"""
    
    def test_basic_reshape(self):
        """Test basic reshaping"""
        x = np.zeros((2, 3, 4))
        output_structure = ["a", "b", "c"]
        shape_dict = {"a": 2, "b": 3, "c": 4}
        
        result = reshape_for_output(x, output_structure, shape_dict)
        self.assertEqual(result, [2, 3, 4])
        
    def test_tuple_reshape(self):
        """Test reshaping with tuples (merging)"""
        x = np.zeros((2, 3, 4))
        output_structure = ["a", ("b", "c")]
        shape_dict = {"a": 2, "b": 3, "c": 4}
        
        result = reshape_for_output(x, output_structure, shape_dict)
        self.assertEqual(result, [2, 12])
        
    def test_numeric_reshape(self):
        """Test reshaping with numeric values"""
        x = np.zeros((2, 3, 4))
        output_structure = ["a", ("2", "b")]
        shape_dict = {"a": 2, "b": 3}
        
        result = reshape_for_output(x, output_structure, shape_dict)
        self.assertEqual(result, [2, 6])
        
    def test_wildcard_reshape(self):
        """Test reshaping with wildcards"""
        x = np.zeros((2, 3, 4, 5))
        output_structure = ["a", "*", "b"]
        shape_dict = {"a": 2, "b": 5, "*": [3, 4]}
        
        result = reshape_for_output(x, output_structure, shape_dict)
        self.assertEqual(result, [2, 3, 4, 5])
        
    def test_infer_dimension(self):
        """Test inferring a missing dimension"""
        x = np.zeros((2, 3, 4))  # 24 elements
        output_structure = ["a", "d"]
        shape_dict = {"a": 2}
        
        result = reshape_for_output(x, output_structure, shape_dict)
        self.assertEqual(result, [2, 12])
        
    def test_error_multiple_unknown(self):
        """Test error with multiple unknown dimensions"""
        x = np.zeros((2, 3, 4))
        output_structure = ["d", "e"]  # Both unknown
        shape_dict = {}
        
        with self.assertRaises(RearrangeError):
            reshape_for_output(x, output_structure, shape_dict)


class TestRearrange(unittest.TestCase):
    """Tests for the main rearrange function"""
    
    def test_basic_transpose(self):
        """Test basic dimension transposition"""
        x = np.zeros((2, 3, 4))
        result = rearrange(x, "a b c -> c a b")
        
        self.assertEqual(result.shape, (4, 2, 3))
        
    def test_merge_dimensions(self):
        """Test merging dimensions"""
        x = np.zeros((2, 3, 4))
        result = rearrange(x, "a b c -> a (b c)")
        
        self.assertEqual(result.shape, (2, 12))
        
    def test_combine_operations(self):
        """Test combining split, merge and transpose"""
        x = np.zeros((6, 8))
        result = rearrange(x, "(a b) (c d) -> c (b d) a", a=2, b=3, c=4, d=2)
        
        self.assertEqual(result.shape, (4, 6, 2))
        
    def test_wildcard_dimensions(self):
        """Test using wildcard dimensions"""
        x = np.zeros((2, 3, 4, 5))
        result = rearrange(x, "a * b -> b * a")
        
        self.assertEqual(result.shape, (5, 3, 4, 2))
        
    def test_ellipsis_dimensions(self):
        """Test using ellipsis dimensions"""
        x = np.zeros((2, 3, 4, 5, 6))
        result = rearrange(x, "a ... b -> b ... a")
        
        self.assertEqual(result.shape, (6, 3, 4, 5, 2))
        
    def test_numeric_dimensions(self):
        """Test using numeric dimensions"""
        x = np.zeros((6, 4))
        result = rearrange(x, "(a 2) b -> 2 (a b)", a=3)
        
        self.assertEqual(result.shape, (2, 12))
        
    def test_non_numpy_error(self):
        """Test error with non-numpy input"""
        x = [1, 2, 3]
        
        with self.assertRaises(TypeError):
            rearrange(x, "a -> a")
            
    def test_pattern_format_error(self):
        """Test error with invalid pattern format"""
        x = np.zeros((2, 3))
        
        with self.assertRaises(RearrangeError):
            rearrange(x, "a b")  # Missing arrow
            
    def test_unused_args_error(self):
        """Test error with unused shape dict arguments"""
        x = np.zeros((2, 3))
        
        with self.assertRaises(RearrangeError):
            rearrange(x, "a b -> b a", c=4)  # c is unused
            
    def test_dimension_mismatch_error(self):
        """Test error with dimension count mismatch"""
        x = np.zeros((2, 3))
        
        with self.assertRaises(RearrangeError):
            rearrange(x, "a b c -> a b c")  # Input has only 2 dims
            
    def test_product_mismatch_error(self):
        """Test error with product mismatch"""
        x = np.zeros((5, 3))
        
        with self.assertRaises(RearrangeError):
            rearrange(x, "(a b) c -> (a c) b", a=2, b=3)  # 2*3 = 6 != 5
            
    def test_real_world_examples(self):
        """Test some real-world examples"""
        # Convert batch-channel-height-width to batch-height-width-channel (BCHW -> BHWC)
        x = np.zeros((32, 3, 128, 128))
        result = rearrange(x, "b c h w -> b h w c")
        self.assertEqual(result.shape, (32, 128, 128, 3))
        
        # Matrix multiplication via einsum-like notation
        x = np.random.rand(10, 5)
        y = np.random.rand(5, 7)
        result = rearrange(np.einsum('ij,jk->ik', x, y), "i j -> i j")
        self.assertEqual(result.shape, (10, 7))
        
    def test_performance_large_array(self):
        """Test performance with a large array (not strictly a unit test)"""
        x = np.zeros((32, 64, 64, 3))  # Typical image batch
        result = rearrange(x, "b h w c -> b c h w")
        self.assertEqual(result.shape, (32, 3, 64, 64))


if __name__ == "__main__":
    unittest.main()
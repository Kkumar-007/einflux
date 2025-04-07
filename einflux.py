# import numpy as np
# import re


# class RearrangeError(Exception):
#     """Exception raised for errors in the rearrange function."""
#     pass


# def parse_pattern(pattern):
#     """Parse a pattern string into a structure more efficiently."""
#     tokens = []
#     current = ""
#     stack = []
#     bracket_level = 0
    
#     for char in pattern:
#         if char == '(' and bracket_level == 0:
#             if current:
#                 tokens.append(current)
#                 current = ""
#             bracket_level = 1
#         elif char == ')' and bracket_level == 1:
#             if current:
#                 stack.append(current)
#                 current = ""
#             tokens.append(tuple(stack))
#             stack = []
#             bracket_level = 0
#         elif char == ' ':
#             if current:
#                 if bracket_level == 0:
#                     tokens.append(current)
#                 else:
#                     stack.append(current)
#                 current = ""
#         else:
#             current += char
    
#     if current:
#         tokens.append(current)
    
#     return tokens


# def expand_dims(shape, structure, shape_dict, merging=False):
#     """Expand dimensions based on the pattern structure - optimized version."""
#     expanded_shape = []
#     shape_idx = 0
#     wildcard_dims = []
    
#     # Process structure in a single pass
#     for item in structure:
#         if item in ('*', '...'):
#             # For wildcards, calculate remaining dimensions in one operation
#             remaining_explicit_dims = sum(1 for x in structure if x not in ('*', '...'))
#             wildcard_dim_count = len(shape) - remaining_explicit_dims
            
#             # Store wildcard dimensions directly
#             wildcard_start = shape_idx
#             wildcard_end = shape_idx + wildcard_dim_count
#             wildcard_dims = shape[wildcard_start:wildcard_end]
#             expanded_shape.extend(wildcard_dims)
#             shape_idx = wildcard_end
#         elif isinstance(item, tuple):
#             if merging:
#                 # For merging, calculate product in one operation
#                 dim_slice = shape[shape_idx:shape_idx + len(item)]
#                 if dim_slice:
#                     expanded_shape.append(np.prod(dim_slice))
#                     shape_idx += len(item)
#             else:
#                 # For splitting, validate and add individual dimensions
#                 if shape_idx >= len(shape):
#                     raise RearrangeError("Not enough dimensions in input shape")
                    
#                 dim_size = shape[shape_idx]
#                 shape_idx += 1
                
#                 # Calculate product directly
#                 product = np.prod([shape_dict.get(sub, 1) for sub in item if not sub.isdigit()])
#                 numeric_product = np.prod([int(sub) for sub in item if sub.isdigit()], initial=1)
#                 product *= numeric_product
                
#                 if product != dim_size:
#                     raise RearrangeError(f"Product of dimensions {item} ({product}) doesn't match input shape {dim_size}")
                
#                 # Add each sub-dimension
#                 expanded_shape.extend([shape_dict[sub] if not sub.isdigit() else int(sub) for sub in item])
#         else:
#             # Regular dimension
#             if shape_idx < len(shape):
#                 expanded_shape.append(shape[shape_idx])
#                 if item not in ('_'):
#                     shape_dict[item] = shape[shape_idx]
#                 shape_idx += 1
#             elif item in shape_dict:
#                 expanded_shape.append(shape_dict[item])
#             else:
#                 raise RearrangeError(f"Not enough dimensions in tensor for pattern, missing '{item}'")

#     # Store wildcard dimensions
#     if wildcard_dims:
#         if '*' in structure:
#             shape_dict['*'] = wildcard_dims
#         elif '...' in structure:
#             shape_dict['...'] = wildcard_dims

#     return expanded_shape


# def get_permutation(input_structure, output_structure, shape_dict):
#     """Get the permutation order from input to output structure - optimized."""
#     # Create mappings from dimension names to positions in a single pass
#     input_dims = {}
#     expanded_input = []
    
#     # Process input structure
#     pos = 0
#     for item in input_structure:
#         if item in ('*', '...') and item in shape_dict:
#             wildcard_dims = shape_dict[item]
#             for i, _ in enumerate(wildcard_dims):
#                 dim_name = f"{item}_{i}"
#                 input_dims[dim_name] = pos
#                 expanded_input.append(dim_name)
#                 pos += 1
#         elif isinstance(item, tuple):
#             for dim in item:
#                 input_dims[dim] = pos
#                 expanded_input.append(dim)
#                 pos += 1
#         else:
#             input_dims[item] = pos
#             expanded_input.append(item)
#             pos += 1
    
#     # Process output structure to build permutation directly
#     permutation = []
#     for item in output_structure:
#         if item in ('*', '...') and item in shape_dict:
#             wildcard_dims = shape_dict[item]
#             for i, _ in enumerate(wildcard_dims):
#                 dim_name = f"{item}_{i}"
#                 if dim_name in input_dims:
#                     permutation.append(input_dims[dim_name])
#         elif isinstance(item, tuple):
#             for dim in item:
#                 if dim in input_dims:
#                     permutation.append(input_dims[dim])
#         else:
#             if item in input_dims:
#                 permutation.append(input_dims[item])
    
#     return permutation


# def reshape_for_output(x, output_structure, shape_dict):
#     """Determine final shape for output tensor - optimized."""
#     final_shape = []
    
#     for item in output_structure:
#         if item in ('*', '...') and item in shape_dict:
#             # Add wildcard dimensions directly
#             final_shape.extend(shape_dict[item])
#         elif isinstance(item, tuple):
#             # Fast path for merging dimensions
#             has_numeric = any(sub.isdigit() for sub in item)
            
#             if has_numeric:
#                 # Process numeric factors once
#                 numeric_product = np.prod([int(sub) for sub in item if sub.isdigit()], initial=1)
#                 # Process non-numeric factors
#                 named_product = np.prod([shape_dict.get(sub, 1) for sub in item if not sub.isdigit()], initial=1)
#                 final_shape.append(numeric_product * named_product)
#             else:
#                 # Simple merge - get product directly
#                 product = np.prod([shape_dict[sub] for sub in item if sub in shape_dict], initial=1)
#                 final_shape.append(product)
#         else:
#             # Regular dimension
#             if item in shape_dict:
#                 final_shape.append(shape_dict[item])
#             else:
#                 final_shape.append(-1)  # Will infer from data
    
#     # Infer missing dimensions efficiently
#     unknown_count = final_shape.count(-1)
    
#     if unknown_count == 1:
#         # Calculate unknown dimension in one operation
#         known_product = np.prod([dim for dim in final_shape if dim != -1], initial=1)
#         unknown_dim = x.size // known_product
#         final_shape = [unknown_dim if dim == -1 else dim for dim in final_shape]
#     elif unknown_count > 1:
#         raise RearrangeError("Cannot infer multiple unknown dimensions.")
    
#     return final_shape


# def rearrange(x, pattern, **user_shape_dict):
#     """
#     Rearranges the dimensions of a tensor according to the pattern - optimized version.
#     """
#     if not isinstance(x, np.ndarray):
#         raise TypeError("Input must be a NumPy array.")

#     # Split pattern once
#     try:
#         input_pat, output_pat = pattern.split('->')
#         input_pat = input_pat.strip()
#         output_pat = output_pat.strip()
#     except ValueError:
#         raise RearrangeError("Pattern must contain '->' separator.")

#     # Parse patterns
#     input_structure = parse_pattern(input_pat)
#     output_structure = parse_pattern(output_pat)

#     # Gather all variables in a single pass
#     used_vars = set()
#     for structure in (input_structure, output_structure):
#         for item in structure:
#             if isinstance(item, tuple):
#                 used_vars.update(sub for sub in item if not sub.isdigit())
#             elif item not in ('*', '...'):
#                 used_vars.add(item)

#     # Validate shape_dict arguments
#     unused = set(user_shape_dict) - used_vars
#     if unused:
#         raise RearrangeError(f"Unused shape_dict arguments: {unused}")

#     # Initialize shape dictionary with user values
#     shape_dict = {k: v for k, v in user_shape_dict.items() if k in used_vars}
    
#     # Infer dimensions from input in a single pass
#     shape_pos = 0
#     for item in input_structure:
#         if isinstance(item, tuple):
#             if shape_pos < len(x.shape):
#                 # Process product in one operation
#                 product = np.prod([
#                     int(sub) if sub.isdigit() else shape_dict.get(sub, 1) 
#                     for sub in item
#                 ], initial=1)
                
#                 if product != x.shape[shape_pos]:
#                     raise RearrangeError(f"Product of {item} ({product}) doesn't match input shape {x.shape[shape_pos]}")
                
#                 # Infer unknown dimensions
#                 for sub in item:
#                     if not sub.isdigit() and sub not in shape_dict:
#                         # Try to infer this dimension
#                         remaining_product = np.prod([
#                             int(other) if other.isdigit() else shape_dict.get(other, 1)
#                             for other in item if other != sub
#                         ], initial=1)
                        
#                         shape_dict[sub] = x.shape[shape_pos] // remaining_product
            
#             shape_pos += 1
#         elif item not in ('*', '...'):
#             if item not in shape_dict and shape_pos < len(x.shape):
#                 shape_dict[item] = x.shape[shape_pos]
#             shape_pos += 1

#     # Skip dimension validation if wildcards are present
#     if '*' not in input_structure and '...' not in input_structure:
#         explicit_dims = len(input_structure)
#         if explicit_dims != len(x.shape):
#             raise RearrangeError(
#                 f"{'Expected' if len(x.shape) > explicit_dims else 'Input tensor has'} "
#                 f"{len(x.shape)} dimensions but pattern {'expects' if len(x.shape) > explicit_dims else 'has'} {explicit_dims}."
#             )

#     # Combined reshape and transform operations
#     try:
#         # Step 1: Reshape for splits (more efficient implementation)
#         new_shape = expand_dims(list(x.shape), input_structure, shape_dict, merging=False)
#         x = x.reshape(new_shape)
        
#         # Step 2: Transpose dimensions (only if needed)
#         permute_order = get_permutation(input_structure, output_structure, shape_dict)
#         if permute_order and permute_order != list(range(len(permute_order))):
#             x = np.transpose(x, permute_order)
        
#         # Step 3: Final reshape (merges, repeats)
#         final_shape = reshape_for_output(x, output_structure, shape_dict)
        
#         # Verify shapes match
#         if np.prod(x.shape) != np.prod(final_shape):
#             raise RearrangeError(f"Shape mismatch: input has {np.prod(x.shape)} elements, output would have {np.prod(final_shape)}")
        
#         return x.reshape(final_shape)
        
#     except Exception as e:
#         if isinstance(e, RearrangeError):
#             raise
#         # Provide more specific error messages
#         if "reshape" in str(e).lower():
#             raise RearrangeError(f"Reshape error: {str(e)}")
#         elif "transpose" in str(e).lower():
#             raise RearrangeError(f"Transpose error: {str(e)}")
#         else:
#             raise RearrangeError(f"Error during rearrangement: {str(e)}")

import numpy as np
import re


class RearrangeError(Exception):
    """Custom exception for tensor reshaping errors"""
    pass


def parse_pattern(pattern):
    """Parse a pattern string like "a b (c d)" into a structured list"""
    result = []
    current_token = ""
    group_tokens = []
    in_parens = 0
    
    # Walk through the pattern character by character
    for char in pattern:
        # Opening parenthesis - start a group
        if char == '(' and in_parens == 0:
            if current_token:  # Save any token we were building
                result.append(current_token)
                current_token = ""
            in_parens = 1
        # Closing parenthesis - finish the group
        elif char == ')' and in_parens == 1:
            if current_token:  # Add the last token in the group
                group_tokens.append(current_token)
                current_token = ""
            result.append(tuple(group_tokens))  # Add the whole group as a tuple
            group_tokens = []  # Reset for next group
            in_parens = 0
        # Space - token separator
        elif char == ' ':
            if current_token:
                if in_parens:
                    group_tokens.append(current_token)
                else:
                    result.append(current_token)
                current_token = ""
        # Any other character - add to current token
        else:
            current_token += char
    
    # Don't forget the last token if there is one
    if current_token:
        result.append(current_token)
    
    return result


def expand_dims(shape, structure, dims_dict, merging=False):
    """
    Expands the dimensions based on the pattern structure.
    
    Args:
        shape: The input tensor shape
        structure: Parsed structure from the pattern
        dims_dict: Dictionary of known dimension sizes
        merging: Whether we're merging or splitting dimensions
    
    Returns:
        List of expanded dimensions
    """
    result = []
    idx = 0
    wildcard_dimensions = []
    
    for item in structure:
        # Handle wildcards (* and ...)
        if item in ('*', '...'):
            # Figure out how many dimensions the wildcard represents
            non_wildcard_dims = sum(1 for x in structure if x not in ('*', '...'))
            wildcard_count = len(shape) - non_wildcard_dims
            
            if wildcard_count < 0:
                # This happens when there are not enough dimensions in the tensor
                raise RearrangeError("Input tensor doesn't have enough dimensions for wildcard")
                
            wildcard_dimensions = shape[idx:idx+wildcard_count]
            result.extend(wildcard_dimensions)
            idx += wildcard_count
            
        # Handle dimension groups like (a b)
        elif isinstance(item, tuple):
            if merging:
                # When merging, we combine several dimensions into one
                dims_to_merge = shape[idx:idx+len(item)]
                if dims_to_merge:  # Make sure we have dimensions to merge
                    merged_dim = 1
                    for d in dims_to_merge:
                        merged_dim *= d
                    result.append(merged_dim)
                    idx += len(item)
            else:
                # When splitting, we break one dimension into several
                if idx >= len(shape):
                    raise RearrangeError("Not enough dimensions in input shape")
                    
                current_dim = shape[idx]
                idx += 1
                
                # Calculate the expected product of the split dimensions
                expected_product = 1
                for sub_dim in item:
                    if sub_dim.isdigit():
                        expected_product *= int(sub_dim)
                    else:
                        expected_product *= dims_dict.get(sub_dim, 1)
                
                # Verify the product matches
                if expected_product != current_dim:
                    raise RearrangeError(f"Product of {item} ({expected_product}) doesn't match input dimension {current_dim}")
                
                # Add each split dimension
                for sub_dim in item:
                    if sub_dim.isdigit():
                        result.append(int(sub_dim))
                    else:
                        result.append(dims_dict[sub_dim])
        
        # Handle regular dimensions
        else:
            if idx < len(shape):
                dim_val = shape[idx]
                result.append(dim_val)
                # Store dimension in dict unless it's a placeholder '_'
                if item != '_':
                    dims_dict[item] = dim_val
                idx += 1
            elif item in dims_dict:
                # Use already known dimension
                result.append(dims_dict[item])
            else:
                raise RearrangeError(f"Missing dimension '{item}' - not in input shape or provided values")

    # Store any wildcard dimensions we found
    if wildcard_dimensions:
        if '*' in structure:
            dims_dict['*'] = wildcard_dimensions
        elif '...' in structure:
            dims_dict['...'] = wildcard_dimensions

    return result


def get_permutation(input_struct, output_struct, dims_dict):
    """Figure out how to rearrange dimensions from input to output pattern"""
    # Map dimension names to their positions in expanded input
    dim_positions = {}
    expanded_dims = []
    
    # First pass - build the mapping of dimension names to positions
    pos = 0
    for item in input_struct:
        # Handle wildcards
        if item in ('*', '...') and item in dims_dict:
            wildcard_dims = dims_dict[item]
            for i in range(len(wildcard_dims)):
                dim_key = f"{item}_{i}"  # Create unique keys for each wildcard dimension
                dim_positions[dim_key] = pos
                expanded_dims.append(dim_key)
                pos += 1
        # Handle grouped dimensions
        elif isinstance(item, tuple):
            for sub_dim in item:
                dim_positions[sub_dim] = pos
                expanded_dims.append(sub_dim)
                pos += 1
        # Handle regular dimensions
        else:
            dim_positions[item] = pos
            expanded_dims.append(item)
            pos += 1
    
    # Build the permutation order by mapping output dimensions to their input positions
    permutation = []
    for item in output_struct:
        # Handle wildcards in output
        if item in ('*', '...') and item in dims_dict:
            wildcard_dims = dims_dict[item]
            for i in range(len(wildcard_dims)):
                dim_key = f"{item}_{i}"
                if dim_key in dim_positions:
                    permutation.append(dim_positions[dim_key])
        # Handle grouped dimensions in output
        elif isinstance(item, tuple):
            for sub_dim in item:
                if sub_dim in dim_positions:
                    permutation.append(dim_positions[sub_dim])
        # Handle regular dimensions in output
        else:
            if item in dim_positions:
                permutation.append(dim_positions[item])
    
    return permutation


def reshape_for_output(tensor, output_struct, dims_dict):
    """Calculate the final output shape based on the output pattern"""
    final_shape = []
    
    for item in output_struct:
        # Handle wildcards
        if item in ('*', '...') and item in dims_dict:
            final_shape.extend(dims_dict[item])
            
        # Handle dimension groups
        elif isinstance(item, tuple):
            # Check if any dimension in the group is numeric
            has_numeric = any(sub.isdigit() for sub in item)
            
            if has_numeric:
                # Handle mix of numeric and named dimensions
                numeric_product = 1
                for sub in item:
                    if sub.isdigit():
                        numeric_product *= int(sub)
                
                named_product = 1
                for sub in item:
                    if not sub.isdigit():
                        # Default to 1 if dimension not found (will be inferred)
                        named_product *= dims_dict.get(sub, 1)
                
                final_shape.append(numeric_product * named_product)
            else:
                # Simple product of named dimensions
                product = 1
                for sub in item:
                    if sub in dims_dict:
                        product *= dims_dict[sub]
                final_shape.append(product)
                
        # Handle regular dimensions
        else:
            if item in dims_dict:
                final_shape.append(dims_dict[item])
            else:
                # Mark for inference (might be inferred from total size)
                final_shape.append(-1)
    
    # Try to infer any missing dimensions
    unknown_dims = final_shape.count(-1)
    
    if unknown_dims == 1:
        # We can infer exactly one dimension
        known_product = 1
        for dim in final_shape:
            if dim != -1:
                known_product *= dim
        
        # Total size divided by product of known dimensions
        unknown_size = tensor.size // known_product
        
        # Replace the -1 with the inferred size
        for i in range(len(final_shape)):
            if final_shape[i] == -1:
                final_shape[i] = unknown_size
                break
    elif unknown_dims > 1:
        raise RearrangeError("Can't infer multiple unknown dimensions at once")
    
    return final_shape


def rearrange(x, pattern, **kwargs):
    """
    Rearrange a tensor according to the given pattern.
    
    This function lets you reshape and permute tensor dimensions in one go,
    using an intuitive pattern syntax similar to einops.
    
    Args:
        x: NumPy array to rearrange
        pattern: String pattern like "a b -> b a" or "b c h w -> b (c h w)"
        **kwargs: Known dimension sizes
        
    Returns:
        Rearranged NumPy array
    """
    if not isinstance(x, np.ndarray):
        raise TypeError("Input must be a NumPy array")

    # Split the pattern into input and output parts
    try:
        input_pattern, output_pattern = pattern.split('->')
        input_pattern = input_pattern.strip()
        output_pattern = output_pattern.strip()
    except ValueError:
        raise RearrangeError("Pattern must contain '->' to separate input and output")

    # Parse the patterns
    input_struct = parse_pattern(input_pattern)
    output_struct = parse_pattern(output_pattern)

    # Find all dimension names used in the patterns
    used_dims = set()
    for struct in (input_struct, output_struct):
        for item in struct:
            if isinstance(item, tuple):
                # Only add non-numeric dimensions
                used_dims.update(sub for sub in item if not sub.isdigit())
            elif item not in ('*', '...'):
                used_dims.add(item)

    # Check for unused kwargs
    unused_kwargs = set(kwargs) - used_dims
    if unused_kwargs:
        raise RearrangeError(f"Unused dimension arguments: {unused_kwargs}")

    # Initialize dimension sizes with provided values
    dims_dict = {k: v for k, v in kwargs.items() if k in used_dims}
    
    # Infer dimensions from input tensor
    dim_idx = 0
    for item in input_struct:
        if isinstance(item, tuple):
            if dim_idx < len(x.shape):
                # For dimension groups, check if we can infer any unknown dimensions
                current_dim = x.shape[dim_idx]
                
                # Calculate product of known dimensions in the group
                known_product = 1
                unknown_dims = []
                
                for sub in item:
                    if sub.isdigit():
                        known_product *= int(sub)
                    elif sub in dims_dict:
                        known_product *= dims_dict[sub]
                    else:
                        unknown_dims.append(sub)
                
                # If exactly one unknown dimension, we can infer it
                if len(unknown_dims) == 1:
                    # Safety check to avoid division by zero
                    if known_product == 0:
                        raise RearrangeError(f"Can't infer {unknown_dims[0]} with zero known product")
                    
                    # Infer the unknown dimension
                    dims_dict[unknown_dims[0]] = current_dim // known_product
                    
                    # Verify our inference is correct (no remainder)
                    if dims_dict[unknown_dims[0]] * known_product != current_dim:
                        raise RearrangeError(f"Can't evenly divide {current_dim} by {known_product} for {unknown_dims[0]}")
            
            dim_idx += 1
        elif item not in ('*', '...'):
            # For regular dimensions, just copy the size
            if dim_idx < len(x.shape) and item not in dims_dict:
                dims_dict[item] = x.shape[dim_idx]
            dim_idx += 1

    # Skip dimension count validation if wildcards are present
    has_wildcards = ('*' in input_struct or '...' in input_struct)
    if not has_wildcards:
        # Count explicit dimensions (accounting for tuples)
        explicit_dims = 0
        for item in input_struct:
            if isinstance(item, tuple):
                explicit_dims += 1  # A tuple is one dimension in the input tensor
            else:
                explicit_dims += 1
        
        if explicit_dims != len(x.shape):
            # Better error message that tells you which way the mismatch goes
            if len(x.shape) > explicit_dims:
                raise RearrangeError(f"Input tensor has {len(x.shape)} dimensions but pattern has only {explicit_dims}")
            else:
                raise RearrangeError(f"Pattern expects {explicit_dims} dimensions but input tensor has only {len(x.shape)}")

    try:
        # Step 1: Split dimensions according to input pattern
        new_shape = expand_dims(list(x.shape), input_struct, dims_dict, merging=False)
        reshaped = x.reshape(new_shape)
        
        # Step 2: Rearrange dimensions
        perm = get_permutation(input_struct, output_struct, dims_dict)
        # Only transpose if necessary
        if perm and list(perm) != list(range(len(perm))):
            transposed = np.transpose(reshaped, perm)
        else:
            transposed = reshaped
        
        # Step 3: Merge dimensions according to output pattern
        output_shape = reshape_for_output(transposed, output_struct, dims_dict)
        
        # Make sure the total size is preserved
        if np.prod(transposed.shape) != np.prod(output_shape):
            raise RearrangeError(f"Element count mismatch: {np.prod(transposed.shape)} vs {np.prod(output_shape)}")
        
        # Final reshape
        return transposed.reshape(output_shape)
        
    except Exception as e:
        # Handle different kinds of errors with helpful messages
        if isinstance(e, RearrangeError):
            raise  # Pass through our custom errors
        
        # Provide better context for numpy errors
        error_msg = str(e).lower()
        if "reshape" in error_msg:
            raise RearrangeError(f"Reshape error: {str(e)}")
        elif "transpose" in error_msg:
            raise RearrangeError(f"Transpose error: {str(e)}")
        else:
            raise RearrangeError(f"Error during rearrangement: {str(e)}")
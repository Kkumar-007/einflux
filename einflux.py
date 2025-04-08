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
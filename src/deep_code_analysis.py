"""
Deep code analysis: Compare searchlogit vs SearchLibrium key methods
to identify the source of the 120-point log-likelihood gap.
"""

import difflib
import inspect

# Import both implementations
from searchlogit._choice_model import DiscreteChoiceModel as SG_DCM
from SearchLibrium._choice_model import DiscreteChoiceModel as SL_DCM

print("=" * 100)
print("DEEP CODE COMPARISON: searchlogit vs SearchLibrium")
print("=" * 100)

# List of critical methods to compare
critical_methods = [
    'setup_design_matrix',
    'get_loglik_gradient',
    'arrange_long_format',
    'balance_panels',
]

for method_name in critical_methods:
    print(f"\n{'='*100}")
    print(f"METHOD: {method_name}")
    print(f"{'='*100}")

    try:
        sg_method = getattr(SG_DCM, method_name, None)
        sl_method = getattr(SL_DCM, method_name, None)

        if sg_method is None:
            print(f"Method '{method_name}' not found in searchlogit")
            continue
        if sl_method is None:
            print(f"Method '{method_name}' not found in SearchLibrium")
            continue

        # Get source code
        try:
            sg_source = inspect.getsource(sg_method)
        except:
            sg_source = "Could not retrieve source"

        try:
            sl_source = inspect.getsource(sl_method)
        except:
            sl_source = "Could not retrieve source"

        # Compare line by line
        sg_lines = sg_source.splitlines(keepends=True)
        sl_lines = sl_source.splitlines(keepends=True)

        # Get diff
        diff = list(difflib.unified_diff(
            sl_lines,
            sg_lines,
            fromfile=f'SearchLibrium.{method_name}',
            tofile=f'searchlogit.{method_name}',
            n=2
        ))

        if diff:
            print(f"DIFFERENCES FOUND ({len(diff)} lines):")
            print(''.join(diff[:100]))  # Show first 100 lines
            if len(diff) > 100:
                print(f"... ({len(diff) - 100} more lines)")
        else:
            print("No differences found in this method")

    except Exception as e:
        print(f"Error comparing {method_name}: {str(e)[:100]}")

print("\n" + "="*100)
print("ADDITIONAL: Checking for methods in searchlogit but NOT in SearchLibrium")
print("="*100)

sg_methods = set(dir(SG_DCM))
sl_methods = set(dir(SL_DCM))

sg_only = sg_methods - sl_methods
sl_only = sl_methods - sg_methods

if sg_only:
    print(f"\nMethods in searchlogit but NOT in SearchLibrium ({len(sg_only)}):")
    for method in sorted(sg_only):
        if not method.startswith('_'):
            print(f"  - {method}")

if sl_only:
    print(f"\nMethods in SearchLibrium but NOT in searchlogit ({len(sl_only)}):")
    for method in sorted(sl_only):
        if not method.startswith('_'):
            print(f"  - {method}")

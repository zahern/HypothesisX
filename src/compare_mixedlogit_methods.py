"""
Compare MixedLogit specific methods between searchlogit and SearchLibrium
"""

import difflib
import inspect

from searchlogit.mixed_logit import MixedLogit as SG_MXL
from SearchLibrium.MixedLogit import MixedLogit as SL_MXL

print("=" * 100)
print("COMPARING MixedLogit CLASS METHODS")
print("=" * 100)

# Critical MixedLogit methods
critical_methods = [
    'fit',
    'generate_draws',
    '_jax_mxl_negloglik',
    'get_loglik_gradient',
]

for method_name in critical_methods:
    print(f"\n{'='*100}")
    print(f"METHOD: {method_name}")
    print(f"{'='*100}")

    sg_method = getattr(SG_MXL, method_name, None)
    sl_method = getattr(SL_MXL, method_name, None)

    if sg_method is None:
        print(f"Method '{method_name}' NOT FOUND in searchlogit")
    if sl_method is None:
        print(f"Method '{method_name}' NOT FOUND in SearchLibrium")

    if sg_method and sl_method:
        try:
            sg_source = inspect.getsource(sg_method)
            sl_source = inspect.getsource(sl_method)

            sg_lines = sg_source.splitlines(keepends=True)
            sl_lines = sl_source.splitlines(keepends=True)

            diff = list(difflib.unified_diff(
                sl_lines,
                sg_lines,
                fromfile=f'SearchLibrium.{method_name}',
                tofile=f'searchlogit.{method_name}',
                n=1
            ))

            if diff:
                print(f"DIFFERENCES FOUND ({len(diff)} lines):")
                print(''.join(diff[:150]))
                if len(diff) > 150:
                    print(f"\n... ({len(diff) - 150} more lines)")
            else:
                print("IDENTICAL - No differences found")

        except Exception as e:
            print(f"Error: {str(e)[:100]}")

print("\n" + "="*100)
print("CHECKING METHOD AVAILABILITY")
print("="*100)

sg_methods = set([m for m in dir(SG_MXL) if not m.startswith('__') and callable(getattr(SG_MXL, m))])
sl_methods = set([m for m in dir(SL_MXL) if not m.startswith('__') and callable(getattr(SL_MXL, m))])

sg_only = sg_methods - sl_methods
sl_only = sl_methods - sg_methods

if sg_only:
    print(f"\nIn searchlogit but NOT SearchLibrium ({len(sg_only)}):")
    for m in sorted(sg_only)[:20]:
        print(f"  - {m}")
    if len(sg_only) > 20:
        print(f"  ... and {len(sg_only)-20} more")

if sl_only:
    print(f"\nIn SearchLibrium but NOT searchlogit ({len(sl_only)}):")
    for m in sorted(sl_only)[:20]:
        print(f"  - {m}")
    if len(sl_only) > 20:
        print(f"  ... and {len(sl_only)-20} more")

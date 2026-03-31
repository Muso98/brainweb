# tumor/templatetags/text_filters.py
from django import template

register = template.Library()

@register.filter(name='replace')
def replace(value, args):
    """
    Usage in template: {{ mytext|replace:"old,new" }}
    Example: {{ "whole_tumor"|replace:"_, " }} -> "whole tumor"
    """
    try:
        old, new = args.split(',', 1)
    except ValueError:
        return value
    return str(value).replace(old, new)

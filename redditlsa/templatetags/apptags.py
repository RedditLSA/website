from django import template

register = template.Library()


@register.filter()
def negate(x):
    return -x


@register.filter()
def get_item(lst, i):
    return lst[i]

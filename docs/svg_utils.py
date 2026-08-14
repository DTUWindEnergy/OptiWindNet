"""Shared SVG formatting helpers for documentation artifacts."""

import re
from xml.dom import Node, minidom

SVG_TEXT_ELEMENTS = frozenset(
    {'foreignObject', 'script', 'style', 'text', 'textPath', 'tspan'}
)


def prettify_svg(svg: str) -> str:
    """Format an SVG document with one space per XML nesting level."""
    document = minidom.parseString(svg)

    def indent_element(
        element: minidom.Element,
        depth: int,
        preserve_text: bool = False,
    ) -> None:
        preserve_text = preserve_text or (
            element.getAttribute('xml:space') == 'preserve'
            or element.localName in SVG_TEXT_ELEMENTS
        )
        children = list(element.childNodes)
        structural_children = [
            child
            for child in children
            if child.nodeType
            in (Node.ELEMENT_NODE, Node.COMMENT_NODE, Node.PROCESSING_INSTRUCTION_NODE)
        ]
        has_text = any(
            child.nodeType in (Node.TEXT_NODE, Node.CDATA_SECTION_NODE)
            and child.data.strip()
            for child in children
        )

        for child in children:
            if child.nodeType == Node.ELEMENT_NODE:
                indent_element(child, depth + 1, preserve_text)

        if preserve_text or has_text or not structural_children:
            return

        for child in children:
            if child.nodeType == Node.TEXT_NODE and not child.data.strip():
                element.removeChild(child)
                child.unlink()

        for child in list(element.childNodes):
            element.insertBefore(
                document.createTextNode(f'\n{" " * (depth + 1)}'), child
            )
        element.appendChild(document.createTextNode(f'\n{" " * depth}'))

    root = document.documentElement
    if root is None:
        raise ValueError('SVG output has no document element')
    indent_element(root, 0)
    declaration = re.match(r'\s*(<\?xml\b.*?\?>)', svg, flags=re.IGNORECASE)
    parts = [
        child.toxml()
        for child in document.childNodes
        if child.nodeType != Node.TEXT_NODE or child.data.strip()
    ]
    if declaration:
        parts.insert(0, declaration.group(1))
    return '\n'.join(parts) + '\n'

import svgwrite

# ---- Matplotlib default colors ----
# Matplotlib's default "tab10" palette includes these:
# blue:  '#1f77b4', orange: '#ff7f0e', red: '#d62728'
colors = {
    "infected":  "#d62728",  # red
    "exposed":   "#ff7f0e",  # orange
    "susceptible": "#1f77b4" # blue
}
# -----------------------------------

# SVG icon generator
def make_person_icon(filename: str, line_color: str):
    size = 256
    dwg = svgwrite.Drawing(filename, size=(size, size))

    # Outer circle - outline color configurable
    dwg.add(
        dwg.circle(
            center=(size / 2, size / 2),
            r=size / 2 - 32,
            stroke=line_color,
            fill="white",
            stroke_width=8,
        )
    )

    # Body arc - same color
    dwg.add(
        dwg.path(
            d=f"M {size/2 - 55} {size/2 + 45} A 55 55 0 0 1 {size/2 + 55} {size/2 + 45}",
            stroke=line_color,
            fill="white",
            stroke_width=8,
        )
    )

    # Head - same color
    dwg.add(
        dwg.circle(
            center=(size / 2, size / 2 - 35),
            r=35,
            stroke=line_color,
            fill="white",
            stroke_width=8,
        )
    )

    dwg.save()
    print(f"Saved {filename}")


# ---- Generate three variants ----
make_person_icon("infected_node.svg", colors["infected"])
make_person_icon("exposed_node.svg", colors["exposed"])
make_person_icon("susceptible_node.svg", colors["susceptible"])

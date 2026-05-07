import ezdxf

#==================================================================================================
def generate_complex_hospital(filename="../../Data/FloorPlan.dxf"):
    print("Generating FloorPlan")
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()

    # 0. Create Layers
    doc.layers.add('WALLS', color=7)
    doc.layers.add('DOORS', color=3)
    doc.layers.add('BEDS', color=1)
    doc.layers.add('ROOM_LABELS', color=5)

    # 1. Walls
    walls = [
        ((0, 5), (20, 5)), ((0, 0), (20, 0)),                        # Corridor
        ((0, 5), (0, 15)), ((0, 15), (10, 15)), ((10, 15), (10, 5)), # ICU Ward
        ((10, 15), (20, 15)), ((20, 15), (20, 5))                    # Standard Ward
    ]
    for start, end in walls:
        msp.add_line(start, end, dxfattribs={'layer': 'WALLS'})

    # 2. Doors (Leaving gaps in walls, drawing door lines)
    doors = [
        ((2, 5), (3, 5)),    # ICU Door
        ((12, 5), (13, 5))   # Standard Ward Door
    ]
    for start, end in doors:
        msp.add_line(start, end, dxfattribs={'layer': 'DOORS'})

    # 3. Beds (Rectangles represented as 4 lines)
    beds = [
        # ICU Bed 1
        [((1, 12), (3, 12)), ((3, 12), (3, 14)), ((3, 14), (1, 14)), ((1, 14), (1, 12))],
        # Standard Bed 1
        [((11, 12), (13, 12)), ((13, 12), (13, 14)), ((13, 14), (11, 14)), ((11, 14), (11, 12))]
    ]
    for bed_lines in beds:
        for start, end in bed_lines:
            msp.add_line(start, end, dxfattribs={'layer': 'BEDS'})

    # 4. Room Labels (Placed roughly in the center of the zones)
    msp.add_text("CORRIDOR", dxfattribs={'layer': 'ROOM_LABELS', 'height': 0.5}).set_placement((10, 2))
    msp.add_text("ICU_WARD", dxfattribs={'layer': 'ROOM_LABELS', 'height': 0.5}).set_placement((5, 10))
    msp.add_text("STD_WARD", dxfattribs={'layer': 'ROOM_LABELS', 'height': 0.5}).set_placement((15, 10))

    doc.saveas(filename)
    print(f"Saved to {filename}")
#==================================================================================================

#==================================================================================================
if __name__ == "__main__":
    generate_complex_hospital()
#==================================================================================================

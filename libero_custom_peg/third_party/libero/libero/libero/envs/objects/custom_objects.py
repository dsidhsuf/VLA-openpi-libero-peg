from robosuite.models.objects import BoxObject
from robosuite.models.objects.composite.bin import Bin


class EarbudProtoObject(BoxObject):
    def __init__(self, name="earbud_proto", obj_name="earbud_proto"):
        super().__init__(
            name=name,
            # BoxObject.size = half-size
            # full size = 0.056 x 0.024 x 0.012
            size=(0.028, 0.012, 0.006),
            rgba=(1.0, 0.1, 0.1, 1.0),
            density=700,
            friction=(4.0, 0.03, 0.003),
            joints="default",
            obj_type="all",
        )
        self.category_name = "earbud_proto"
        self.rotation = (0, 0)
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}


class ChargingSlotProtoObject(Bin):
    def __init__(self, name="charging_slot_proto", obj_name="charging_slot_proto"):
        # 目标：竖直的长方形孔，而不是横向槽
        # 细柱竖直插入时，孔口只比它的横截面略大
        wall = 0.004

        # 内孔尺寸（只略大于细柱横截面）
        inner_x = 0.028   # 2.8 cm
        inner_y = 0.016   # 1.6 cm
        inner_z = 0.040   # 孔深 4.0 cm

        outer_x = inner_x + 2 * wall
        outer_y = inner_y + 2 * wall
        outer_z = inner_z

        super().__init__(
            name=name,
            bin_size=(outer_x, outer_y, outer_z),
            wall_thickness=wall,
            transparent_walls=False,
            friction=(1.5, 0.005, 0.0001),
            density=1200.0,
            use_texture=False,
            rgba=(0.18, 0.18, 0.20, 1.0),
        )

        self.category_name = "charging_slot_proto"
        self.rotation = (0, 0)
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}

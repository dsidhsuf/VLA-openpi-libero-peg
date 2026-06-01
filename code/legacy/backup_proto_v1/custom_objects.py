from robosuite.models.objects import CapsuleObject, HollowCylinderObject

class EarbudProtoObject(CapsuleObject):
    def __init__(self, name="earbud_proto", obj_name="earbud_proto"):
        super().__init__(
            name=name,
            size=(0.012, 0.014),   # 更粗更短，便于抓取
            rgba=(0.15, 0.95, 0.20, 1.0),
            density=400,
            friction=(2.0, 0.01, 0.001),
            joints="default",
            obj_type="all",
        )
        self.category_name = "earbud_proto"
        # 让耳机初始平放，而不是容易竖着倒下
        self.rotation = (1.57, 1.57)
        self.rotation_axis = "x"
        self.object_properties = {"vis_site_names": {}}

class ChargingSlotProtoObject(HollowCylinderObject):
    def __init__(self, name="charging_slot_proto", obj_name="charging_slot_proto"):
        super().__init__(
            name=name,
            outer_radius=0.032,
            inner_radius=0.014,
            height=0.035,
            ngeoms=16,
            rgba=(0.25, 0.25, 0.28, 1.0),
            density=800.0,
            friction=(1.0, 0.005, 0.0001),
            make_half=False,
        )
        self.category_name = "charging_slot_proto"
        self.rotation = (0, 0)
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}
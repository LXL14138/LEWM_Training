
BLOCKS = [
    {
        "name": "target",
        "xy": [0.425, 0.0],
        "size": [0.04, 0.04, 0.10],  # 与两个红柱相同：底面 4×4 cm，高 10 cm
        "color": [0.08, 0.75, 0.20],  # RGB：绿色
    },
    {
        "name": "guard_left",
        "xy": [0.425, -0.05],  # 与绿色表面的间隙为 1 cm
        "size": [0.04, 0.04, 0.10],  # 底面 4×4 cm，高 10 cm
        "color": [0.95, 0.15, 0.15],  # RGB：红色
    },
    {
        "name": "guard_right",
        "xy": [0.425, 0.05],  # 与左柱对称，初始位置即为安全位移基准
        "size": [0.04, 0.04, 0.10],  # 底面 4×4 cm，高 10 cm
        "color": [0.95, 0.15, 0.15],
    },
]

# 夹取中心从绿色目标旁上方开始：水平相差 5 cm，高度 16 cm。
ARM_START_XYZ = [0.375, 0.0, 0.16]
# 初始化检查的高度下限；不改变机器人运动范围或随机采集的默认范围。
ARM_START_MIN_Z_M = 0.16

# 世界坐标中的固定正面相机，拉近观察三个柱子；不跟踪机械臂。
# 仅影响 front_pixels；side_pixels 和独立腕部相机的安装保持不变。
# 位置保持之前的近景设置；此次只将视场角从 45° 轻微扩大到 48°。
FRONT_CAMERA_POSITION = [0.70125, 0.0, 0.324]
FRONT_CAMERA_LOOK_AT = [0.425, 0.0, 0.12]
# Slightly wider coverage for guard interaction; same position and look-at.
FRONT_CAMERA_FOVY_DEG = 48.0

# 关闭投影阴影，保留正常光照明暗；所有相机和采集共用此设置。
RENDER_SHADOWS = False

# 夹爪绕世界竖直轴水平旋转 90 度，仍然朝下；全程固定，不允许动作改变。
ARM_FIXED_YAW_DEG = 90.0

# 仅检查两个红色保护柱：相对世界竖直方向倾斜超过此角度，判定 unsafe。
# 这是可调整的任务阈值，不是物理上的必然倾倒角；接触本身不算危险。
UNSAFE_TILT_DEG = 30.0

# 暂停红柱 COM 位移危险判定；仍记录位移，方便以后重新启用。
GUARD_DISPLACEMENT_CHECK_ENABLED = False
# 仅在上方开关为 True 时生效；单位为米，与倾角条件为“或”的关系。
UNSAFE_DISPLACEMENT_M = 0.01

# 绿色块质心相对地面 z=0 至少高 10 cm（不是在初始高度上再加 10 cm）。
# 两侧夹指须真实接触目标，并在此高度连续保持 1 秒仿真时间。
TARGET_COMPLETION_HEIGHT_M = 0.10
TARGET_HOLD_SECONDS = 1.0

# 每侧夹指对绿色块的法向接触力总和必须超过此数值，过滤零力接触。
# 这是仿真中的夹持检测标准，不是严格的稳定抓取/力闭合保证。
GRASP_MIN_NORMAL_FORCE_N = 0.001

# 只影响保存的预览图片；不会影响 world model，因为此脚本没有加载模型。
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# 当前三个物体都是底面 4×4 cm、高 10 cm 的同尺寸长方柱。
# 恢复旧物体布局：绿色 size 改回 [0.04, 0.04, 0.04]，红柱 y 改回 ±0.09；起点/相机另行配置。
# 物理尺寸、中心高度、质量和惯量会一起更新，不是只改变外观。

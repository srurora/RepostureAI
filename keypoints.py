# Define landmark indices for the keypoints
keypoint_indices = {
    "left_elbow": 13,
    "right_elbow": 14,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "nose": 0,
}

# Custom connections (list of tuples with keypoint indices to connect)
custom_connections = [
    (11, 23),   # left shoulder and left hip
    (12, 24),   # right shoulder and right hip
    (11, 13),  # Left shoulder to left elbow
    (13, 15),  # Left elbow to left wrist
    (12, 14),  # Right shoulder to right elbow
    (14, 16),  # Right elbow to right wrist
    (23, 25),  # Left hip to left knee
    (25, 27),  # Left knee to left ankle
    (24, 26),  # Right hip to right knee
    (26, 28),  # Right knee to right ankle
    (11, 12),  # Left shoulder to right shoulder
    (23, 24),  # Left hip to right hip
]


joint_triplets = {
        "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
        "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
        "left_shoulder": ("left_hip", "left_shoulder", "left_elbow"),
        "right_shoulder": ("right_hip", "right_shoulder", "right_elbow"),
        "left_hip": ("left_shoulder", "left_hip", "left_knee"),
        "right_hip": ("right_shoulder", "right_hip", "right_knee"),
        "left_knee": ("left_hip", "left_knee", "left_ankle"),
        "right_knee": ("right_hip", "right_knee", "right_ankle"),
        "middle_hip": ("right_knee", "middle_hip", "left_knee")
    }
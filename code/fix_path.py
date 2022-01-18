import os
def fix_data_path():
  root="/content/drive/MyDrive/Multi-task_Engagement_Detection/data/OpeFace/OpenFace_features/"
  for case in ["Train","validation"]:
    for face_feature in os.listdir(root + case):
      if "txt" not in face_feature:
        os.rename(os.path.join(root + case,face_feature),os.path.join(root + case,face_feature+".txt"))

if __name__ == "__main__":
    fix_data_path()
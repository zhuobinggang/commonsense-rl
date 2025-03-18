## 2025.3.11改动

* 重要：此前的Game对象存在reset之后没有清空recipe的问题，可能对模型性能造成重大影响。因此进行了重新实验。
* 重要：此前的模型以steps_limit = 30进行了训练和测试，这可能会限制模型的性能，因此进行了重新实验（不限长度训练 + 50步valid + 99步test）。

训练好的模型文件夹： `/home/taku/Downloads/cog2019_ftwp/trained_models/behavior_clone_0311/`


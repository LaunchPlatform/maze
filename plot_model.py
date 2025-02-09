import json

import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()

accuracy_vals = []
loss_vals = []
with (
    open("random_model.json", "rt") as random_fo,
    open("handcraft_model0.json", "rt") as handcraft_fo,
    open("handcraft_model_with_maxpool_upscale.json", "rt") as upscale_fo,
):
    for random_line, handcraft_line, upscale_line in zip(
        random_fo.readlines(),
        handcraft_fo.readlines(),
        upscale_fo.readlines(),
    ):
        random_row = json.loads(random_line)
        handcraft_row = json.loads(handcraft_line)
        upscale_row = json.loads(upscale_line)

        accuracy_vals.append(
            (
                random_row["accuracy"],
                handcraft_row["accuracy"],
                upscale_row["accuracy"],
            )
        )
        loss_vals.append(
            (
                random_row["loss"],
                handcraft_row["loss"],
                upscale_row["loss"],
            )
        )

ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.plot(accuracy_vals)
ax1.tick_params(axis="y")
ax1.legend(
    ["Random Accuracy", "Handcraft Accuracy", "MaxPoolUpscale Accuracy"],
    loc="center right",
    bbox_to_anchor=(0.5, 0.3, 0.5, 0.5),
)


ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

ax2.set_ylabel("Loss")  # we already handled the x-label with ax1
ax2.plot(loss_vals, linestyle="--")
ax2.tick_params(axis="y")
ax2.legend(
    ["Random Loss", "Handcraft Loss", "MaxPoolUpscale Loss"],
    loc="center right",
    bbox_to_anchor=(0.5, 0.1, 0.5, 0.5),
)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

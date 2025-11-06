#!/usr/bin/env python3

"""
This script modifies an existing ONNX face detection model (specifically,
an SCRFD-based model like 'buffalo_detector.onnx') by appending a
post-processing graph.

The original model outputs raw predictions (scores, bbox deltas, kps deltas)
from multiple FPN (Feature Pyramid Network) levels. This script adds the
necessary ONNX operators to:
1.  Decode these raw outputs into absolute bounding box (x1, y1, x2, y2)
    and keypoint (x, y) coordinates by applying stride and anchor information.
2.  Concatenate the decoded predictions from all FPN levels.
3.  Add a 'NonMaxSuppression' (NMS) operator to filter a large number of
    predictions down to a small set of final detections.
4.  Make the NMS 'iou_threshold' and 'score_threshold' parameters dynamic
    graph inputs, allowing them to be set at runtime.
5.  Gather the final boxes, scores, and keypoints based on the NMS results.

This process embeds the post-processing logic directly into the ONNX graph,
simplifying deployment and inference pipelines.
"""

import onnx
import onnx_graphsurgeon as gs
import numpy as np

# --- 1. Configuration & Model Parameters ---
# Define the core parameters of the SCRFD model.
# These values MUST match the architecture of the input ONNX file.

# The strides of the FPN levels (e.g., 8, 16, 32).
STRIDES = [8, 16, 32]
# The number of anchors per location (typically 2 for SCRFD).
NUM_ANCHORS = 2
# The number of FPN levels (Feature Map Count).
FMC = 3
# The input image shape (Height, Width) the model was trained on.
INPUT_SHAPE = (640, 640)  # (H, W)
# Whether the model includes keypoint (landmark) prediction.
USE_KPS = True

# --- I/O File Paths ---
# IMPORTANT: Path to the original ONNX model file.
MODEL_FILE = "models/buffalo_detector.onnx"
# Path to save the new model with embedded post-processing.
OUTPUT_FILE = "models/scfrd_640_with_postprocessing_and_dynamic_thresholding.onnx"

# IMPORTANT: These are the names of the raw output tensors from the original model.
# The order is critical and must match the FPN levels (s8, s16, s32).
# This order was determined by inspecting the model in Netron.
OUTPUT_NAMES = [
    # Scores (s8, s16, s32)
    '448', '471', '494',
    # BBoxes (s8, s16, s32)
    '451', '474', '497',
    # KPS (s8, s16, s32)
    '454', '477', '500'
]

# --- 2. Anchor Generation ---

def get_anchors(input_height, input_width, feat_stride_fpn, num_anchors):
    """
    Generates anchor centers for all FPN levels.

    This pre-computes the (x, y) coordinates of the center of every
    anchor box on all feature maps.

    Args:
        input_height (int): The height of the model's input tensor.
        input_width (int): The width of the model's input tensor.
        feat_stride_fpn (list[int]): List of FPN strides.
        num_anchors (int): Number of anchors per grid cell.

    Returns:
        list[np.ndarray]: A list where each element is an array of
                          anchor centers [N, 2] for a specific FPN level.
    """
    all_anchors = []
    for stride in feat_stride_fpn:
        # Calculate feature map dimensions
        height = input_height // stride
        width = input_width // stride

        # Create a grid of anchor centers (x, y)
        anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)

        # Scale centers by the stride to map them back to input image coordinates
        anchor_centers = (anchor_centers * stride).reshape((-1, 2))

        # Duplicate centers for each anchor box (if num_anchors > 1)
        if num_anchors > 1:
            anchor_centers = np.stack([anchor_centers] * num_anchors, axis=1).reshape((-1, 2))
        
        all_anchors.append(anchor_centers)
    return all_anchors

print("Pre-computing anchors...")
precomputed_anchors = get_anchors(INPUT_SHAPE[0], INPUT_SHAPE[1], STRIDES, NUM_ANCHORS)


# --- 3. Graph-Building Helper Function ---

def create_decoding_nodes(graph, anchors_const, scores_in, bbox_preds_in, kps_preds_in, stride_val):
    """
    Constructs and appends ONNX nodes to decode outputs for a single FPN level.

    This function implements the "distance2bbox" and "distance2kps" logic
    to convert raw model outputs (which are deltas) into absolute
    coordinates.

    Args:
        graph (gs.Graph): The ONNX graph surgeon graph.
        anchors_const (gs.Constant): Constant node containing anchors [N, 2] for this level.
        scores_in (gs.Variable): Input tensor for scores [N, 1].
        bbox_preds_in (gs.Variable): Input tensor for bbox deltas [N, 4].
        kps_preds_in (gs.Variable): Input tensor for kps deltas [N, 10] or None.
        stride_val (int): The stride value for this FPN level.

    Returns:
        (gs.Variable, gs.Variable, gs.Variable):
            - scores_squeezed: Decoded scores [N].
            - decoded_boxes: Decoded bboxes [N, 4] in (x1, y1, x2, y2) format.
            - decoded_kps: Decoded keypoints [N, 10] or None.
    """
    # Create a constant node for the stride value
    stride_const = gs.Constant(f"stride_{stride_val}", values=np.array([stride_val], dtype=np.float32))

    # --- 1. Decode Bounding Boxes (distance2bbox) ---
    
    # Scale bbox predictions (deltas) by the stride
    # bbox_preds_scaled = bbox_preds_in * stride
    bbox_preds_scaled = gs.Variable(f"bbox_preds_scaled_{stride_val}", dtype=np.float32)
    node_bbox_mul = gs.Node(op="Mul",
                            inputs=[bbox_preds_in, stride_const],
                            outputs=[bbox_preds_scaled])
    graph.nodes.append(node_bbox_mul)

    # Split anchor centers [N, 2] into x [N, 1] and y [N, 1]
    # We use Gather(axis=1, index=0) -> Unsqueeze(axes=[1])
    
    # Gather 'x' (anchor center x) - index 0
    anchors_x_gathered = gs.Variable(f"anchors_x_gathered_{stride_val}", dtype=np.float32)
    node_gather_ax = gs.Node(op="Gather", attrs={"axis": 1},
                             inputs=[anchors_const,
                                     gs.Constant(f"gather_ax_idx_{stride_val}", values=np.array(0, dtype=np.int64))],
                             outputs=[anchors_x_gathered])
    # Unsqueeze [N] -> [N, 1]
    anchors_x = gs.Variable(f"anchors_x_{stride_val}", dtype=np.float32)
    node_us_ax = gs.Node(op="Unsqueeze", attrs={"axes": [1]}, inputs=[anchors_x_gathered], outputs=[anchors_x])

    # Gather 'y' (anchor center y) - index 1
    anchors_y_gathered = gs.Variable(f"anchors_y_gathered_{stride_val}", dtype=np.float32)
    node_gather_ay = gs.Node(op="Gather", attrs={"axis": 1},
                             inputs=[anchors_const,
                                     gs.Constant(f"gather_ay_idx_{stride_val}", values=np.array(1, dtype=np.int64))],
                             outputs=[anchors_y_gathered])
    # Unsqueeze [N] -> [N, 1]
    anchors_y = gs.Variable(f"anchors_y_{stride_val}", dtype=np.float32)
    node_us_ay = gs.Node(op="Unsqueeze", attrs={"axes": [1]}, inputs=[anchors_y_gathered], outputs=[anchors_y])
    
    graph.nodes.extend([
        node_gather_ax, node_us_ax,
        node_gather_ay, node_us_ay
    ])

    # Split scaled bbox deltas [N, 4] into l, t, r, b [N, 1 each]
    # This is done using the same Gather + Unsqueeze pattern 
    # (We can get rid of this in the new opset version using "Slice" operator but that 
    # was not supported since the original model was exported with Opset verison <= 11)

    # Gather 'l' (left) - index 0
    bbox_l_gathered = gs.Variable(f"bbox_l_gathered_{stride_val}", dtype=np.float32)
    node_gather_l = gs.Node(op="Gather", attrs={"axis": 1},
                            inputs=[bbox_preds_scaled,
                                    gs.Constant(f"gather_l_idx_{stride_val}", values=np.array(0, dtype=np.int64))],
                            outputs=[bbox_l_gathered])
    bbox_l = gs.Variable(f"bbox_l_{stride_val}", dtype=np.float32)
    node_us_l = gs.Node(op="Unsqueeze", attrs={"axes": [1]}, inputs=[bbox_l_gathered], outputs=[bbox_l])

    # Gather 't' (top) - index 1
    bbox_t_gathered = gs.Variable(f"bbox_t_gathered_{stride_val}", dtype=np.float32)
    node_gather_t = gs.Node(op="Gather", attrs={"axis": 1},
                            inputs=[bbox_preds_scaled,
                                    gs.Constant(f"gather_t_idx_{stride_val}", values=np.array(1, dtype=np.int64))],
                            outputs=[bbox_t_gathered])
    bbox_t = gs.Variable(f"bbox_t_{stride_val}", dtype=np.float32)
    node_us_t = gs.Node(op="Unsqueeze", attrs={"axes": [1]}, inputs=[bbox_t_gathered], outputs=[bbox_t])

    # Gather 'r' (right) - index 2
    bbox_r_gathered = gs.Variable(f"bbox_r_gathered_{stride_val}", dtype=np.float32)
    node_gather_r = gs.Node(op="Gather", attrs={"axis": 1},
                            inputs=[bbox_preds_scaled,
                                    gs.Constant(f"gather_r_idx_{stride_val}", values=np.array(2, dtype=np.int64))],
                            outputs=[bbox_r_gathered])
    bbox_r = gs.Variable(f"bbox_r_{stride_val}", dtype=np.float32)
    node_us_r = gs.Node(op="Unsqueeze", attrs={"axes": [1]}, inputs=[bbox_r_gathered], outputs=[bbox_r])

    # Gather 'b' (bottom) - index 3
    bbox_b_gathered = gs.Variable(f"bbox_b_gathered_{stride_val}", dtype=np.float32)
    node_gather_b = gs.Node(op="Gather", attrs={"axis": 1},
                            inputs=[bbox_preds_scaled,
                                    gs.Constant(f"gather_b_idx_{stride_val}", values=np.array(3, dtype=np.int64))],
                            outputs=[bbox_b_gathered])
    bbox_b = gs.Variable(f"bbox_b_{stride_val}", dtype=np.float32)
    node_us_b = gs.Node(op="Unsqueeze", attrs={"axes": [1]}, inputs=[bbox_b_gathered], outputs=[bbox_b])

    graph.nodes.extend([
        node_gather_l, node_us_l,
        node_gather_t, node_us_t,
        node_gather_r, node_us_r,
        node_gather_b, node_us_b
    ])
    
    # --- Decode BBox Coordinates ---
    # x1 = anchor_x - (delta_l * stride)
    x1 = gs.Variable(f"x1_{stride_val}", dtype=np.float32)
    node_sub_x1 = gs.Node(op="Sub", inputs=[anchors_x, bbox_l], outputs=[x1])
    # y1 = anchor_y - (delta_t * stride)
    y1 = gs.Variable(f"y1_{stride_val}", dtype=np.float32)
    node_sub_y1 = gs.Node(op="Sub", inputs=[anchors_y, bbox_t], outputs=[y1])
    # x2 = anchor_x + (delta_r * stride)
    x2 = gs.Variable(f"x2_{stride_val}", dtype=np.float32)
    node_add_x2 = gs.Node(op="Add", inputs=[anchors_x, bbox_r], outputs=[x2])
    # y2 = anchor_y + (delta_b * stride)
    y2 = gs.Variable(f"y2_{stride_val}", dtype=np.float32)
    node_add_y2 = gs.Node(op="Add", inputs=[anchors_y, bbox_b], outputs=[y2])
    graph.nodes.extend([node_sub_x1, node_sub_y1, node_add_x2, node_add_y2])

    # Re-form the boxes: Concat [N, 1] x 4 -> [N, 4]
    decoded_boxes = gs.Variable(f"decoded_boxes_{stride_val}", dtype=np.float32)
    node_concat_boxes = gs.Node(op="Concat", attrs={"axis": 1},
                                 inputs=[x1, y1, x2, y2],
                                 outputs=[decoded_boxes])
    graph.nodes.append(node_concat_boxes)

    # --- 2. Decode Scores ---
    # Squeeze scores from [N, 1] to [N] for easier concatenation later.
    scores_squeezed = gs.Variable(f"scores_squeezed_{stride_val}", dtype=np.float32)
    node_squeeze_scores = gs.Node(op="Squeeze", attrs={"axes": [1]},
                                  inputs=[scores_in],
                                  outputs=[scores_squeezed])
    graph.nodes.append(node_squeeze_scores)


    # --- 3. Decode Keypoints (distance2kps) ---
    decoded_kps = None
    if USE_KPS and kps_preds_in is not None:
        # Keypoint predictions must also be scaled by the stride.
        kps_preds_scaled = gs.Variable(f"kps_preds_scaled_{stride_val}", dtype=np.float32)
        node_kps_mul = gs.Node(op="Mul",
                               inputs=[kps_preds_in, stride_const],
                               outputs=[kps_preds_scaled])
        graph.nodes.append(node_kps_mul)

        # Decode KPS: kps = anchor_center + (kps_delta * stride)
        # We need to tile anchor centers [N, 2] to match kps shape [N, 10]
        # Tiling [N, 2] with repeats [1, 5] results in [N, 10] (x,y,x,y,...)
        tiled_anchors = gs.Variable(f"tiled_anchors_{stride_val}", dtype=np.float32)
        node_tile_anchors = gs.Node(op="Tile",
                                     inputs=[anchors_const,
                                             gs.Constant(f"tile_repeats_{stride_val}", values=np.array([1, 5], dtype=np.int64))],
                                     outputs=[tiled_anchors])
        graph.nodes.append(node_tile_anchors)

        # decoded_kps = tiled_anchors + kps_preds_scaled
        decoded_kps = gs.Variable(f"decoded_kps_{stride_val}", dtype=np.float32)
        node_add_kps = gs.Node(op="Add",
                               inputs=[tiled_anchors, kps_preds_scaled],
                               outputs=[decoded_kps])
        graph.nodes.append(node_add_kps)

    return scores_squeezed, decoded_boxes, decoded_kps


# --- 4. Load and Prepare the Original Graph ---
print(f"Loading original model: {MODEL_FILE}")
# Load the original ONNX model
graph = gs.import_onnx(onnx.load(MODEL_FILE))

# Get a reference to all original graph outputs
model_outputs = [t for t in graph.outputs]
# Create a map for easy lookup by name
graph_outputs_map = {o.name: o for o in model_outputs}

# Detach the original output tensors from the graph.
# We will add new outputs at the end of our post-processing chain.
for out in graph.outputs:
    out.outputs.clear()
print("Detached original outputs.")

# --- 5. Iterate FPN Levels and Add Decoding Nodes ---
# These lists will store the final decoded tensors from each level.
decoded_scores_list = []
decoded_boxes_list = []
decoded_kps_list = []

print("Adding decoding nodes for all FPN levels...")
for i in range(FMC):
    # Get the raw output tensors for this level using the pre-defined names
    scores_in = graph_outputs_map[OUTPUT_NAMES[i]]
    bbox_preds_in = graph_outputs_map[OUTPUT_NAMES[i + FMC]]
    
    kps_preds_in = None
    if USE_KPS:
        kps_preds_in = graph_outputs_map[OUTPUT_NAMES[i + FMC * 2]]
    
    # Create an ONNX Constant node for this level's precomputed anchors
    anchors_const = gs.Constant(f"anchors_s{STRIDES[i]}",
                                values=precomputed_anchors[i])
    
    # Call the helper function to build the decoding graph for this level
    scores_out, boxes_out, kps_out = create_decoding_nodes(
        graph,
        anchors_const,
        scores_in,
        bbox_preds_in,
        kps_preds_in,
        STRIDES[i]
    )
    
    # Store the resulting decoded tensors
    decoded_scores_list.append(scores_out)
    decoded_boxes_list.append(boxes_out)
    if kps_out:
        decoded_kps_list.append(kps_out)

print("Finished adding decoding nodes.")

# --- 6. Concatenate Results from all Levels ---
# Combine the decoded tensors from all FPN levels into three single tensors
# (one for scores, one for boxes, one for keypoints).

# Concat all scores: [N1], [N2], [N3] -> [N_total]
all_scores = gs.Variable("all_scores_decoded", dtype=np.float32)
node_concat_scores = gs.Node(op="Concat", attrs={"axis": 0},
                             inputs=decoded_scores_list,
                             outputs=[all_scores])
graph.nodes.append(node_concat_scores)

# Concat all boxes: [N1, 4], [N2, 4], [N3, 4] -> [N_total, 4]
all_boxes = gs.Variable("all_boxes_decoded", dtype=np.float32)
node_concat_boxes = gs.Node(op="Concat", attrs={"axis": 0},
                             inputs=decoded_boxes_list,
                             outputs=[all_boxes])
graph.nodes.append(node_concat_boxes)

all_kps = None
if USE_KPS and decoded_kps_list:
    # Concat all kps: [N1, 10], [N2, 10], [N3, 10] -> [N_total, 10]
    all_kps = gs.Variable("all_kps_decoded", dtype=np.float32)
    node_concat_kps = gs.Node(op="Concat", attrs={"axis": 0},
                               inputs=decoded_kps_list,
                               outputs=[all_kps])
    graph.nodes.append(node_concat_kps)

print("Added concatenation nodes.")

# --- 7. Prepare Tensors for NonMaxSuppression ---
# The ONNX NonMaxSuppression operator requires specific tensor shapes
# which include a batch_size dimension. We assume batch_size = 1.

# Reshape boxes from [N_total, 4] to [1, N_total, 4]
boxes_for_nms = gs.Variable("boxes_for_nms", dtype=np.float32)
node_unsqueeze_boxes_nms = gs.Node(op="Unsqueeze", attrs={"axes": [0]},
                                   inputs=[all_boxes],
                                   outputs=[boxes_for_nms])
graph.nodes.append(node_unsqueeze_boxes_nms)

# Reshape scores from [N_total] to [1, 1, N_total] (batch_size=1, num_classes=1)
scores_for_nms = gs.Variable("scores_for_nms", dtype=np.float32)
node_unsqueeze_scores_nms = gs.Node(op="Unsqueeze", attrs={"axes": [0, 1]},
                                    inputs=[all_scores],
                                    outputs=[scores_for_nms])
graph.nodes.append(node_unsqueeze_scores_nms)


# --- 8. Add the NonMaxSuppression Node ---
# Add the NMS node to filter overlapping boxes.

# --- Define NMS Parameters ---
# Constant for max boxes to output per class
max_output_boxes_per_class = gs.Constant("max_output_boxes", values=np.array([200], dtype=np.int64))

# --- Define Dynamic Threshold Inputs ---
# We make the thresholds graph inputs so they can be changed at runtime
# without modifying the model again...

# Create a new graph input for the IoU threshold
iou_threshold_input = gs.Variable(name="iou_threshold_input", dtype=np.float32, shape=[1])
# Create a new graph input for the score threshold
score_threshold_input = gs.Variable(name="score_threshold_input", dtype=np.float32, shape=[1])

# Add these new inputs to the graph's main input list
graph.inputs.extend([iou_threshold_input, score_threshold_input])

# --- Create the NMS Node ---
# Output 'selected_indices' has shape [num_selected_indices, 3]
# Each row is [batch_index, class_index, box_index]
selected_indices = gs.Variable("selected_indices", dtype=np.int64)
node_nms = gs.Node(op="NonMaxSuppression",
                   inputs=[boxes_for_nms,
                           scores_for_nms,
                           max_output_boxes_per_class,
                           iou_threshold_input,   # Use new dynamic input
                           score_threshold_input], # Use new dynamic input
                   outputs=[selected_indices])
graph.nodes.append(node_nms)
print("Added NonMaxSuppression node with dynamic thresholds.")

# --- 9. Gather Final Detections Using NMS Indices ---
# The 'selected_indices' tensor from NMS is used to gather the
# final surviving boxes, scores, and keypoints.

# 'selected_indices' has shape [num_selected, 3].
# Use Gather to select this column.
box_indices_gathered = gs.Variable("box_indices_gathered", dtype=np.int64)
node_gather_indices = gs.Node(op="Gather", attrs={"axis": 1},
                              inputs=[selected_indices,
                                      gs.Constant("gather_indices_idx", values=np.array(2, dtype=np.int64))],
                              outputs=[box_indices_gathered])
graph.nodes.append(node_gather_indices)
# 'box_indices_gathered' now has shape [num_selected], containing the
# indices of the boxes to keep from the 'all_boxes' tensor.

# Gather final boxes:
# Select rows from 'all_boxes' [N_total, 4] using 'box_indices_gathered' [num_selected]
final_boxes = gs.Variable("final_boxes", dtype=np.float32)
node_gather_boxes = gs.Node(op="Gather", attrs={"axis": 0},
                            inputs=[all_boxes, box_indices_gathered],
                            outputs=[final_boxes])
graph.nodes.append(node_gather_boxes)

# Gather final scores:
# Select rows from 'all_scores' [N_total] using 'box_indices_gathered' [num_selected]
final_scores = gs.Variable("final_scores", dtype=np.float32)
node_gather_scores = gs.Node(op="Gather", attrs={"axis": 0},
                             inputs=[all_scores, box_indices_gathered],
                             outputs=[final_scores])
graph.nodes.append(node_gather_scores)

# This list will hold the final, user-facing output tensors
final_outputs = [final_boxes, final_scores]

if all_kps is not None:
    # Gather final keypoints (if they exist)
    final_kps = gs.Variable("final_kps", dtype=np.float32)
    node_gather_kps = gs.Node(op="Gather", attrs={"axis": 0},
                              inputs=[all_kps, box_indices_gathered],
                              outputs=[final_kps])
    graph.nodes.append(node_gather_kps)
    final_outputs.append(final_kps)

print("Added final Gather nodes.")

# --- 10. Set New Graph Outputs and Cleanup ---

# Set the graph's outputs to our new final tensors
graph.outputs = final_outputs

# Clean up any disconnected nodes and perform a topological sort
graph.cleanup().toposort()
print("Graph cleanup complete.")

# --- 11. Export the Modified Model ---

# Set a compatible opset version (12 is widely supported and includes NMS) (MOST IMPORTANT)
graph.opset = 12

# Export the graph surgeon object back to an ONNX model
onnx_model = gs.export_onnx(graph)

# Save the model
onnx.save(onnx_model, OUTPUT_FILE)
print(f"Successfully saved new model to: {OUTPUT_FILE}")
print("You can now run inference on this new model.")

import numpy as np
import cv2 as cv
import dearpygui.dearpygui as dpg

from rpd import viewport_properties
import camera_main
from rpd import filtering
from rpd import features
ui_elements : dict = {
    "contours_count" : 0,
    "display_mode" : "contours",
    "approximate contours" : True
}

diplay_modes: list = ["original", "filtered", "contours"]

def process_frame(frame):

    img = frame
    img_1 = filtering.canny_amax_adaptive_filter(img)
    #Find contours with opencv
    contours, hierarchy = cv.findContours(img_1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = features.contours_filter_small_area(contours, viewport_properties.FEATURES_FILTER_MIN_AREA)
    contours = features.contours_filter_solidity(contours, viewport_properties.FEATURES_FILTER_SOLIDITY)
    if dpg.get_value(ui_elements["approximate contours"]):
        contours = features.approx_polygon_from_contour(contours)
    contours = features.contours_filter_vertices(contours)
    contours = features.contours_filter_positional_2(contours, viewport_properties.FEATURES_FILTER_POSITIONAL_2_DISTANCE)
    #Draw contours
    img_2 = np.zeros((img_1.shape[0], img_1.shape[1], 3), dtype=np.uint8)
    if dpg.get_value(ui_elements["display_mode"]) == "filtered":
        img_2 = cv.cvtColor(img_1, cv.COLOR_GRAY2BGR)
    elif dpg.get_value(ui_elements["display_mode"]) == "original":
        img_2 = img
    img_2 = cv.drawContours(img_2, contours, -1, (0,255,0), 3)
    if dpg.get_value(ui_elements["approximate contours"]):
        for contour in contours:
            M = cv.moments(contour)
            if M["m00"] == 0:
                continue
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            cv.putText(img_2, str(len(contour)), center, cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    #resize to 900x600
    img_2 = cv.resize(img_2, (900, 600))
    #Display image
    cv.imshow("contours", img_2)

    #Display cropped parts of the image
    imgs = features.contours_crop_and_reverse_perspective(frame, contours)
    if len(imgs) != 0:
        # convert all images to 100x100
        imgs = [cv.resize(img, (100, 100)) for img in imgs]
        # stack vertically
        imgs = np.vstack(imgs)
        #display image
        cv.imshow("cropped", imgs)
    dpg.set_value(ui_elements["contours_count"], "Contours count: "+str(len(contours)))
    if cv.waitKey(1) == ord('q'):
       return

dpg.create_context()
dpg.create_viewport(width=600, height=600)
dpg.setup_dearpygui()
with dpg.window(label="Contour detection", width=600, height=600, no_resize=True, no_move=True, no_collapse=True, no_close=True):
    ui_elements['contours_count'] = dpg.add_text("Contours count: "+str(ui_elements["contours_count"]))
    ui_elements['display_mode'] = dpg.add_combo(label = "Display mode", items=diplay_modes, default_value=ui_elements["display_mode"])
    ui_elements['approximate contours'] = dpg.add_checkbox(label="Approximate contours", default_value=ui_elements["approximate contours"])

dpg.show_viewport()
camera = camera_main.camera_main_coroutine(process_frame)
while dpg.is_dearpygui_running():
    next(camera)
    dpg.render_dearpygui_frame()

dpg.destroy_context()
import camera_main
import filtering
import features
import debug
import numpy as np
import cv2 as cv
import dearpygui.dearpygui as dpg


ui_elements : dict = {
    "contours_count" : 0,
}

def process_frame(frame):

    img = frame
    img = cv.resize(img, (500, 400))
    img_1 = filtering.canny_amax_adaptive_filter(img)
    #Find contours with opencv
    contours, hierarchy = cv.findContours(img_1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = features.contours_filter_small_area(contours, 200)
    contours = features.contours_filter_solidity(contours, 0.9)
    contours = features.contours_filter_positional_2(contours, 1)
    #Draw contours
    #img_2 = np.zeros((img_1.shape[0], img_1.shape[1], 3), dtype=np.uint8)
    img_2 = cv.cvtColor(img_1, cv.COLOR_GRAY2BGR)
    img_2 = cv.drawContours(img_2, contours, -1, (0,255,0), 3)
    #resize to 900x600
    img_2 = cv.resize(img_2, (900, 600))
    #Display image
    cv.imshow("contours", img_2)
    dpg.set_value(ui_elements["contours_count"], "Contours count: "+str(len(contours)))
    if cv.waitKey(1) == ord('q'):
       return

dpg.create_context()
dpg.create_viewport(width=600, height=600)
dpg.setup_dearpygui()
with dpg.window(label="Contour detection", width=600, height=600, no_resize=True, no_move=True, no_collapse=True, no_close=True):
    ui_elements['contours_count'] = dpg.add_text("Contours count: "+str(ui_elements["contours_count"]))

dpg.show_viewport()
camera = camera_main.camera_main_coroutine(process_frame)
while dpg.is_dearpygui_running():
    next(camera)
    dpg.render_dearpygui_frame()

dpg.destroy_context()
import numpy as np
import cv2
from tracker import Tracker
import time
import imageio
images = []

import numpy as np
import cv2 as cv
import math


def createimage(w,h):
	size = (w, h, 1)
	img = np.ones((w,h,3),np.uint8)*255
	return img

def main():
	data = np.array(np.load('Detections.npy'))[0:10,0:150,0:150]
	tracker = Tracker(150, 30)
	skip_frame_count = 0
	track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
					(127, 127, 255), (255, 0, 255), (255, 127, 255),
					(127, 0, 255), (127, 0, 127),(127, 10, 255), (0,255, 127)]

	for i in range(data.shape[1]):
		centers = data[:,i,:]
		np.random.shuffle(centers)
		frame = createimage(512,512)
		if (len(centers) > 0):
			tracker.update(centers)
			for j in range(len(tracker.tracks)):
				if (len(tracker.tracks[j].trace) > 1):
					x = int(tracker.tracks[j].trace[-1][0,0])
					y = int(tracker.tracks[j].trace[-1][0,1])
					tl = (x-10,y-10)
					br = (x+10,y+10)
					cv2.rectangle(frame,tl,br,track_colors[j],1)
					cv2.putText(frame,str(tracker.tracks[j].trackId), (x-10,y-20),0, 0.5, track_colors[j],2)
					for k in range(len(tracker.tracks[j].trace)):
						x = int(tracker.tracks[j].trace[k][0,0])
						y = int(tracker.tracks[j].trace[k][0,1])
						cv2.circle(frame,(x,y), 3, track_colors[j],-1)
					cv2.circle(frame,(x,y), 6, track_colors[j],1)
				cv2.circle(frame,(int(data[j,i,0]),int(data[j,i,1])), 3, (0,0,0),-1)
			cv2.imshow('image',frame)
			# cv2.imwrite("image"+str(i)+".jpg", frame)
			# images.append(imageio.imread("image"+str(i)+".jpg"))
			time.sleep(0.1)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				cv2.destroyAllWindows()
				break

	# imageio.mimsave('Multi-Object-Tracking.gif', images, duration=0.08)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         

def mpt():
    tracker = Tracker(dist_threshold=150, max_frame_skipped=30, max_trace_length= 20)
    total_time = 0.0
    total_frames = 0
    colours = np.random.rand(32, 3) * 255 #used only for display
    colours = colours.astype(int)
    cap = cv.VideoCapture('inverse_FPV Racing España 2017 - Spain Drone Team_clipped_video.mp4')
    init_feature_params = dict(maxCorners=0, qualityLevel=0.1, minDistance=2 , blockSize=3)
    feature_params = dict(maxCorners=720, qualityLevel=0.0455, minDistance=2 , blockSize=5)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.02))
    height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    video_fps = cap.get(cv.CAP_PROP_FPS)
    print(height, width, video_fps)
    tracks = []
    track_len = 90
    frame_idx = 0
    detect_interval = 1
    green = (0, 255, 0)
    red = (0, 0, 255)
    blue = (255, 0, 0)
    non_homography_point = []
    detections = []
    bboxsize = 15
    # Set the new dimensions
    new_width = 1280
    new_height = int(new_width * height / width)
    print(new_height, new_width)
    tr=[]

    # out = cv.VideoWriter("reslut_MPT_15fps.mp4", cv.VideoWriter_fourcc('D', 'I', 'V', 'X'), video_fps/2,
    #                     (int(new_width), int(new_height)), True)

    while True:
        dets = []
        detection = set()
        ret, frame = cap.read()
        if ret:
            # Resize the image
            frame = cv.resize(frame, (new_width, new_height))
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            vis = frame.copy()
            timer = cv.getTickCount()
            if len(tracks)>0:
                img0 ,img1 = prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1,1,2)
                # 上一帧的角点和当前帧的图像作为输入来得到角点在当前帧的位置  
                p1, st, err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                # 反向检查,当前帧跟踪到的角点及图像和前一帧的图像作为输入来找到前一帧的角点位置  
                p0r, _, _ = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                # 得到角点回溯与前一帧实际角点的位置变化关系 
                d = abs(p0-p0r).reshape(-1,2).max(-1)
                #判断d内的值是否小于1，大于1跟踪被认为是错误的跟踪点
                good = d < 1.0
                new_tracks = []
                for i, (tr, (x, y), flag) in enumerate(zip(tracks, p1.reshape(-1, 2), good)):
                    # 判断是否为正确的跟踪点
                    if not flag:
                        continue
                    # 存储动态的角点
                    tr.append((x, y))
                    # 只保留track_len长度的数据，消除掉前面的超出的轨迹
                    if len(tr) > track_len:
                        del tr[0]
                    # 保存在新的list中
                    new_tracks.append(tr)
                    # cv.circle(vis, (int(x), int(y)), 1, (0, 255, 255), 1)
                
                # for i, (tr, (x, y), flag) in enumerate(zip(tracks, p0.reshape(-1, 2), np.logical_not(good))):
                #     if not flag:
                #         continue
                #     cv.circle(vis, (int(x), int(y)), 3, (255, 0, 255), 1)
                
                # Find homography
                good = good.reshape(-1,1)
                good_new = p1[good == 1]
                # print(p1.shape)
                # print(good_new.shape)
                good_old = p0r[good == 1]
                # print(len(good_old))
                if len(good_old) > 4 and len(good_new) > 4:
                    H_matrix, mask = cv.findHomography(good_old, good_new, cv.USAC_MAGSAC, 
                                                    ransacReprojThreshold = 1.0, maxIters = 2000, confidence = 0.99)
                    # Use homography
                    for point, H_good in zip(good_new, mask[:,0]):
                    # for point, H_good in zip(good_old, np.logical_not(mask[:,0])):
                    # for point, H_good in zip(good_old, mask[:,0]):
                        # print(point)
                        x, y = point
                        # print(x, y)
                        cv.circle(vis, (int(x), int(y)), 3, (green, blue)[H_good], 1)
                        if H_good == 0:
                            non_homography_point.append((int(x), int(y)))
                            dets.append((x, y))
                            # dets.append((int(x-bboxsize), int(y-bboxsize), int(x+bboxsize), int(y+bboxsize)))
                            # detections.append([x,y,7,7,1.0])
                            # f.write(f"{frame_idx},-1,{x},{y},7,7,0.9,-1,-1,-1\n")
                # 更新特征点    
                tracks = new_tracks
                # #以上一振角点为初始点，当前帧跟踪到的点为终点,画出运动轨迹
                # cv.polylines(vis, [np.int32(tr) for tr in tracks], False, green , 1)
                # print('tracks:', len(tracks))
                # print(tracks)
            fps = cv.getTickFrequency() / (cv.getTickCount() - timer)
            # 每隔 detect_interval 时间检测一次特征点
            if frame_idx % detect_interval==0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                if frame_idx !=0:
                    for x,y in [np.int32(tr[-1]) for tr in tracks]:
                        cv.circle(mask, (x, y), 7, 0, -1)
                # if frame_idx < 2:
                #     p = cv.goodFeaturesToTrack(frame_gray, mask=mask, **init_feature_params)
                # else:
                #     p = cv.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)

                p = cv.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1,2):
                        tracks.append([(x, y)])
                        # dets.append((x, y))
                        # dets.append((int(x-bboxsize), int(y-bboxsize), int(x+bboxsize), int(y+bboxsize)))
                        cv.circle(vis, (int(x), int(y)), 3, red, 1)

            if len(non_homography_point) > 0:
                # pass
                # print(np.array(dets))
                tracker.update(np.array(dets))
                for j in range(len(tracker.tracks)):
                    if (len(tracker.tracks[j].trace) > 1):
                        index = int(j % 32)
                        col = colours[index,:].astype(int)
                        color = ( int (col[0]), int (col[1]), int (col[2])) 
                        x = int(tracker.tracks[j].trace[-1][0,0])
                        y = int(tracker.tracks[j].trace[-1][0,1])
                        tl = (x-5,y-5)
                        br = (x+5,y+5)
                        cv.rectangle(vis,tl,br,color,1)
                        cv.putText(vis,str(tracker.tracks[j].trackId), (x-10,y-20),0, 0.5, color,2)
                        # cv.polylines(vis, [np.int32(tr) for tr in tracker.tracks[j].trace], False, green , 1)
                        # for k in range(len(tracker.tracks[j].trace)):
                        #     x = int(tracker.tracks[j].trace[k][0,0])
                        #     y = int(tracker.tracks[j].trace[k][0,1])
                        # #     cv.circle(vis,(x,y), 1, color,-1)
                        #     tr.append((x, y))
                        # # print('tr:', len(tr))
                        # # tr = np.array(tr, dtype=np.int32)
                        # # print(np.int32(tr))
                        # cv.polylines(vis, np.int32([tr]), False, green , 1)
                        cv.circle(vis,(x,y), 7, color,1)
                    # cv.circle(vis,(int(data[j,i,0]),int(data[j,i,1])), 3, (0,0,0),-1)


                # print('non_homography_point:', len(non_homography_point), 'non_homography_point:',non_homography_point)


            text = "track count: %d" % len(tracks)
            cv.putText(vis, text, (0,13), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
            text = "frame count: %d" % frame_idx
            cv.putText(vis, text, (0,26), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
            text = "non_homography_point: %d" % len(non_homography_point)
            cv.putText(vis, text, (0,39), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
            text = "FPS :" + str(fps)
            cv.putText(vis, text, (0,52), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
            non_homography_point = []
            detections = []
            frame_idx += 1
            prev_gray = frame_gray
            cv.imshow('track', vis)
            # out.write(vis)
            ch = cv.waitKey(30)
            if ch ==27:
                # cv.imwrite('track.jpg', vis)
                break
        else:
            break

    cv.destroyAllWindows()
    cap.release()
    # out.release()
    print('Done')
    print('frame_idx:', frame_idx)


if __name__ == '__main__':
	# main()
    mpt()
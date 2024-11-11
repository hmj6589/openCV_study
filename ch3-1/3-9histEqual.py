import cv2
import matplotlib.pyplot as plt

fig=plt.figure()
rows=3  # 가로열이 2
cols=2  # 세로열이 2

img=cv2.imread('mistyroad.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)	# 명암 영상으로 변환하고 출력
ax1=fig.add_subplot(rows,cols,1)    # (0,0) 맨 첫번째에 값을 넣겠다
ax1.axis("off")     # 가로축 세로축 표시하지 않겠다
ax1.imshow(gray,cmap='gray')

h=cv2.calcHist([gray],[0],None,[256],[0,256])	# 히스토그램을 구해 출력
print('histogram', h)
ax2=fig.add_subplot(rows,cols,2)    # (0,1) 맨 첫째줄 오른쪽에 값을 넣겠다
ax2.plot(h,color='r',linewidth=1)

equal=cv2.equalizeHist(gray)	# 히스토그램을 평활화하고 출력
ax3=fig.add_subplot(rows,cols,3)    # (1,0) 2번째줄 왼쪽에 값을 넣겠다
ax3.axis("off")
ax3.imshow(equal,cmap='gray')

h1=cv2.calcHist([equal],[0],None,[256],[0,256])	 # 히스토그램을 구해 출력
print('histogram equal1', h1)
ax4=fig.add_subplot(rows,cols,4)
ax4.plot(h,color='r',linewidth=1)

equal2=cv2.equalizeHist(equal)	# 히스토그램을 평활화하고 출력
ax5=fig.add_subplot(rows,cols,3)    # (1,0) 2번째줄 왼쪽에 값을 넣겠다
ax5.axis("off")
ax5.imshow(equal2,cmap='gray')

h2=cv2.calcHist([equal2],[0],None,[256],[0,256])	 # 히스토그램을 구해 출력
print('histogram equal2', h2)
ax6=fig.add_subplot(rows,cols,4)
ax6.plot(h2,color='r',linewidth=1)

plt.show()
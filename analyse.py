import os
from scipy.io.wavfile import read
from scipy.io.wavfile import write
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
#from va_dilated import generate
from vabaseline2 import generate
from mpl_toolkits.mplot3d import Axes3D
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

def standardize(fs,x,length,file_name0,file_name1):
   M=128
   th = 0.00005
   out = np.zeros((length,1))
   x = x-np.mean(x)
   x = x/np.float((max(abs(x))))
   if len(x)>length:
      j=0
      for i in range(len(x)/M):
         if sum(np.square(x[i*M:(i+1)*M]))/np.float(M)<th:
            continue
         x=x[i*M:len(out)-i*M]
         break
   if len(x)<length:
      out[:len(x),0]=x
   else:
      out = x
   write(file_name0+file_name1,fs,out)
   return np.reshape(out,(-1,))


n_hidden = 20
batch_size = 100

check_dir = sys.argv[1]

length=4096
audio = []
labels = []
fs = -1
length_list =[]
fileName = []
input_dir = "./recordings/"
file_list = os.listdir(input_dir)
for file in file_list:
   if "jackson" in file and not "new" in file:
      fsx, x = read(input_dir+file)
      if fs<0:
         fs = fsx
      if fsx!=fs:
         print("fs missmatching")
         sys.exit()
      x = standardize(fs,x,length,input_dir+"new_",file)
      audio.append(x)
      labels.append(np.int32(file[0]))
      length_list.append(len(x))
      fileName.append(file)

audio = np.array(audio,dtype=np.float32)
length_list = np.asarray(length_list)
fileName = np.asarray(fileName)
labels = np.asarray(labels)

indx = []
for i in range(10):
    tmp = (np.where(labels==i))
    indx.append(tmp[0][:10])
#indx = np.linspace(0,len(audio),batch_size+1,dtype=np.int32).tolist()
#indx = indx[:len(indx)-1]
indx = np.reshape(np.array(indx),(-1))

audio = audio[indx]
labels = labels[indx]
input = tf.placeholder(tf.float32)
output_layer,encoded_layer ,inputx, z_mean , z_stddev= generate(input,length,None,None,n_hidden)



'''input_descriminator = tf.concat((output_layer,tf.reshape(inputx,[-1,length,1])),axis=0)
class_layer2 = tf.layers.conv1d(input_descriminator,64,9,trainable=True,kernel_initializer=tf.random_normal_initializer)
class_layer2 = tf.layers.conv1d(class_layer2,128,9,trainable=True,kernel_initializer=tf.random_normal_initializer)
class_layer2 = tf.layers.conv1d(class_layer2,256,5,trainable=True,kernel_initializer=tf.random_normal_initializer)
class_layer2 = tf.layers.flatten(class_layer2)
class_layer2 = tf.layers.dense(class_layer2,50,trainable=True,kernel_initializer=tf.random_normal_initializer)
class_layer2 = tf.layers.dense(class_layer2,50,trainable=True,kernel_initializer=tf.random_normal_initializer)
class_layer2 = tf.layers.dense(class_layer2,50,trainable=True,kernel_initializer=tf.random_normal_initializer)
class_output2 = tf.layers.dense(class_layer2,1,activation=tf.nn.sigmoid,trainable=True,kernel_initializer=tf.random_normal_initializer)
'''
class_layer = tf.layers.dense(encoded_layer,10,activation=tf.nn.relu,trainable=True,kernel_initializer=tf.random_normal_initializer)
class_layer = tf.layers.dense(class_layer,10,activation=tf.nn.sigmoid,trainable=True,kernel_initializer=tf.random_normal_initializer)
class_layer = tf.layers.dense(class_layer,10,activation=tf.nn.sigmoid,trainable=True,kernel_initializer=tf.random_normal_initializer)
class_output = tf.nn.softmax(class_layer)




out_dir = sys.argv[1] + "/generated_tmp/"
try:
   os.makedirs(out_dir)
except:
   out_dir = sys.argv[1] + "/generated_tmp"+str(random.randint(0,100))+ "/" 
   os.makedirs(out_dir)

check_dir = sys.argv[1] + "/checkpoints/best_model.ckpt"
saver = tf.train.Saver()

sess = tf.Session()
saver.restore(sess,check_dir)

output ,encoded = sess.run([output_layer,encoded_layer],feed_dict={input:audio})

print("output length: " + str(len(output)))
print("encoded shape: "+ str(encoded.shape))

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
X = pca.fit_transform(encoded)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(10):
    indx_i = np.where(labels==i)
    ax.scatter(X[indx_i][:,0],X[indx_i][:,1],X[indx_i][:,2],marker="$ {} $".format(str(i)),label=str(i))
    ax.legend()
    plt.savefig(out_dir+"scatter_pca3d")

pca = PCA(n_components=2)
X = pca.fit_transform(encoded)
plt.figure()
for i in range(10):
    indx_i = np.where(labels==i)
    plt.scatter(X[indx_i][:,0],X[indx_i][:,1],marker="$ {} $".format(str(i)),label=str(i))
    plt.legend()
    plt.savefig(out_dir+"scatter_pca2d")


from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)
X = tsne.fit_transform(encoded)
plt.figure()
for i in range(10):
    indx_i = np.where(labels==i)
    plt.scatter(X[indx_i][:,0],X[indx_i][:,1],marker="$ {} $".format(str(i)),label=str(i))
    plt.legend()
    plt.savefig(out_dir+"scatter_tsne2d")

tsne = TSNE(n_components=3)
X = tsne.fit_transform(encoded)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(10):
    indx_i = np.where(labels==i)
    ax.scatter(X[indx_i][:,0],X[indx_i][:,1],X[indx_i][:,2],marker="$ {} $".format(str(i)),label=str(i))
    ax.legend()
    plt.savefig(out_dir+"scatter_tsne3d")

import scipy

dist = np.zeros((10,10))
eps = np.finfo(np.float32).eps
for i in range(10):
  indx1 = np.where(labels==i)
  mean1 = np.mean(encoded[indx1],axis=0)
  out = sess.run(output_layer,feed_dict={encoded_layer:np.reshape(mean1,(-1,len(mean1)))})
  write(out_dir+"mean"+str(i)+".wav",fs,np.reshape(out,(length,1)))
  cova1 = np.cov(encoded[indx1],rowvar=False)
  for j in range(10):
    if j==i:
       dist[i,j]=0
       continue
    indx2 = np.where(labels==j)
    mean2 = np.mean(encoded[indx2],axis=0)
    cova2 = np.cov(encoded[indx2],rowvar=False)
    cova_sum = 0.5*(cova1+cova2)
    '''if np.linalg.det(cova1) < eps or np.linalg.det(cova2) < eps:
       bhattacharyya[i,j] =0
       continue
    '''
    ##print(np.sqrt(np.sum(np.square(np.mean(encoded[indx2],axis=0)-np.mean(encoded[indx1],axis=0)))))
    #print(np.sqrt(np.sum((mean1-mean2)**2)))
    #print(np.var(encoded[indx2],axis=0))
    dist[i,j] = np.sqrt(abs(np.transpose(mean1-mean2).dot(np.sqrt(abs(np.linalg.inv(cova1).dot(np.linalg.inv(cova2))))).dot(mean1-mean2)))
    #dist[i,j] = dist[i,j] + 1.0/4*np.log(np.linalg.det(cova_sum)/np.sqrt(np.linalg.det(cova1)*np.linalg.det(cova2)))
    #bhattacharyya[i,j] = dist
    #dist[i,j] = np.sqrt(np.transpose(mean1-mean2).dot(np.sqrt(abs(np.linalg.inv(cova1)*np.linalg.inv(cova2)))).dot(mean1-mean2))
    #dist[i,j] = scipy.stats.wasserstein_distance(encoded[0][indx1],encoded[indx2])
    #sys.stdout.write("calculating dist: " + str(i*10+j)+"\r")
    #sys.stdout.flush()
#print(dist)


try:
  os.makedirs(out_dir+"preds/")
except:
  a=0
for i in range(len(output)):
    write(out_dir+"preds/"+str(labels[i])+"_"+str(i)+".wav",fs,output[i])


coord_ini = sess.run([encoded_layer],feed_dict={input:audio[np.where(labels==6)[0][0]]})[0]
coord_fin = sess.run([encoded_layer],feed_dict={input:audio[np.where(labels==6)[0][-1]]})[0]
multiLinspace = np.zeros((100,coord_ini.shape[1]))
for j in range(coord_ini.shape[1]):
         linspace = np.linspace(coord_ini[0][j],coord_fin[0][j],100)
         multiLinspace[:,j]=linspace

out_newdir = out_dir+"from6to6/"
out3= sess.run([ output_layer],feed_dict={encoded_layer:multiLinspace})
try:
  os.makedirs(out_newdir)
except:
  a=0
for i in range(len(out3[0])):
    write( out_newdir+str(i)+'.wav', fs, np.reshape(np.asarray(out3[0][i]),(length,1)) )


coord_ini = sess.run([encoded_layer],feed_dict={input:audio[np.where(labels==6)[0][0]]})[0]
coord_fin = sess.run([encoded_layer],feed_dict={input:audio[np.where(labels==8)[0][0]]})[0]
multiLinspace = np.zeros((100,coord_ini.shape[1]))
for j in range(coord_ini.shape[1]):
         linspace = np.linspace(coord_ini[0][j],coord_fin[0][j],100)
         multiLinspace[:,j]=linspace

out_newdir = out_dir+"from6to8/"
out3= sess.run([ output_layer],feed_dict={encoded_layer:multiLinspace})
try:
  os.makedirs(out_newdir)
except:
  a=0
for i in range(len(out3[0])):
    write( out_newdir+str(i)+'.wav', fs, np.reshape(np.asarray(out3[0][i]),(length,1)) )

wrong_label = []
a=0
from sklearn.neighbors import KNeighborsClassifier
for i in range(len(encoded)):
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(encoded[np.arange(len(encoded))!=i], labels[np.arange(len(encoded))!=i])
    score = (neigh.predict(np.reshape(encoded[i],(1,-1)))==labels[i])
    a = a+score
    if score ==0:
       wrong_label.append(labels[i])
    sys.stdout.write("1NN evaluation: "+str(i)+ "/"+str(len(encoded))+"\r")
    sys.stdout.flush()
print("")
print(a/np.float(len(encoded)))
wrong_label = np.array(wrong_label)
print()
for i in range(10):
   print(sum(wrong_label==i))

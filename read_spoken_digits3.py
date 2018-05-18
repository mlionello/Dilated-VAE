import os
from scipy.io.wavfile import read
from scipy.io.wavfile import write
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from vahybrid import generate
import random
from mpl_toolkits.mplot3d import Axes3D
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

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

epochs = 100000
batch_size = 50
n_hidden = 20
test_size = 50

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

eval_indx = np.linspace(0,len(audio),batch_size+1,dtype=np.int32).tolist()
eval_indx = eval_indx[:len(eval_indx)-1]

audio_eval = np.array(audio,dtype=np.float32)[eval_indx]
label_eval = np.array(labels)[eval_indx]
fileName_eval = np.array(fileName)[eval_indx]


len_audio0 = len(audio)

same_label = []
for i in range(10):
    same_label.append(np.where(np.array(labels) == labels[i])[0])


tmp_labels = np.array(labels)
for i in range(len_audio0):
  for j in range(20):
     audio.append((audio[i]+audio[np.where(tmp_labels == labels[i])[0][j]])/2.0)
     fileName.append(fileName[i])
     labels.append(labels[i])
'''     if j<5:
        audio.append((audio[i]+audio[np.where(tmp_labels == labels[i])[0][j]]+audio[np.where(tmp_labels == labels[i])[0][j+20]])/3.0)
        fileName.append(fileName[i])
        labels.append(labels[i])
        audio.append((audio[i]+audio[np.where(tmp_labels == labels[random.randint(0,9)])[0][j]])/2.0)
        fileName.append(fileName[i])
        labels.append(labels[i])
'''

audio = np.array(audio,dtype=np.float32)
length_list = np.asarray(length_list)
fileName = np.asarray(fileName)
labels = np.asarray(labels)

print("input_shape: " + str(audio.shape))
print("batch_size: " + str(batch_size))


tst_indx = np.linspace(0,len_audio0,test_size+1,dtype=np.int32).tolist()
tst_indx = tst_indx[:len(tst_indx)-1]

tr_indx = []
for i in range(len(audio)):
  if i in tst_indx:
     continue
  tr_indx.append(i)

np.random.shuffle(tr_indx)

training_size = len(tr_indx)

import datetime, os
t = datetime.datetime.now()
t = "%4d%02d%02d-%02d-%02d-%02d" % (t.year,t.month,t.day,t.hour,t.minute,t.second)
t = t+"_gansspokendigits"
dir_check = t+"/checkpoints/"
dir_output = t+"/outputs/"

os.makedirs(t)
os.makedirs(dir_output)
os.makedirs(dir_check)
checkpoint_path = dir_check + "model.ckpt"

training_set=tf.data.Dataset.from_tensor_slices({"audio":audio[tr_indx],"label":labels[tr_indx],"name":fileName[tr_indx]})
testing_set=tf.data.Dataset.from_tensor_slices({"audio":audio[tst_indx],"label":labels[tst_indx],"name":fileName[tst_indx]})

#training_set=training_set.shuffle(buffer_size=training_size*100000)
training_set=training_set.repeat(epochs)
training_set=training_set.batch(batch_size)
iter_train_handle = training_set.make_one_shot_iterator().string_handle()
testing_set=testing_set.repeat(epochs)
testing_set=testing_set.batch(test_size)
iter_val_handle = testing_set.make_one_shot_iterator().string_handle()

handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(
        handle, training_set.output_types, training_set.output_shapes)
next_batch = iterator.get_next()
next_audio = next_batch["audio"]
next_label = next_batch["label"]

next_hotlabel = tf.one_hot(next_label,depth=10)
### generator Input audio junck
istrainable = tf.Variable(True)
output_layer,encoded_layer ,inputx, z_mean , z_stddev= generate(next_audio,length,None,None,n_hidden,istrainable)

### generation loss
generation_loss = tf.reduce_sum(tf.squared_difference(output_layer, inputx), 1)

### posterior probability loss
sigma2=4.
mi2=0.
latent_loss = tf.reduce_sum(tf.log(sigma2)-z_stddev+0.5*tf.square(tf.exp(z_stddev)/sigma2)+0.5*tf.square((z_mean-mi2)/sigma2),1)

### discriminator Input: output Layer
input_descriminator = tf.concat((output_layer,tf.reshape(next_audio,[-1,length,1])),axis=0)
class_layer2 = tf.layers.conv1d(input_descriminator,64,9,trainable=True,kernel_initializer=tf.random_normal_initializer)
class_layer2 = tf.layers.conv1d(class_layer2,128,9,trainable=True,kernel_initializer=tf.random_normal_initializer)
class_layer2 = tf.layers.conv1d(class_layer2,256,5,trainable=True,kernel_initializer=tf.random_normal_initializer)
class_layer2 = tf.layers.flatten(class_layer2)
class_layer2 = tf.layers.dense(class_layer2,50,trainable=True,kernel_initializer=tf.random_normal_initializer)
class_layer2 = tf.layers.dense(class_layer2,50,trainable=True,kernel_initializer=tf.random_normal_initializer)
class_layer2 = tf.layers.dense(class_layer2,50,trainable=True,kernel_initializer=tf.random_normal_initializer)
class_output2 = tf.layers.dense(class_layer2,1,activation=tf.nn.sigmoid,trainable=True,kernel_initializer=tf.random_normal_initializer)
#class_loss_t = tf.multiply(next_hotlabel,tf.divide(tf.log(class_output),tf.log(2.)))
#class_loss_f = tf.multiply(1-next_hotlabel,tf.divide(tf.log(1-class_output),tf.log(2.)))
#class_loss = -tf.reduce_mean(class_loss_t+class_loss_f)
tf_labels = np.concatenate((np.zeros((batch_size,1),dtype=np.float32),np.ones((batch_size,1),dtype=np.float32)),axis=0)
#loss_desc =  tf.multiply(tf_labels,tf.divide(tf.log(class_output2),tf.log(2.)))
loss_desc = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_labels,logits=class_output2)


### classifier in the latent space Input: encoded layer
class_layer = tf.layers.dense(encoded_layer,10,activation=tf.nn.relu,trainable=True,kernel_initializer=tf.random_normal_initializer)
class_layer = tf.layers.dense(class_layer,10,activation=tf.nn.relu,trainable=True,kernel_initializer=tf.random_normal_initializer)
class_layer = tf.layers.dense(class_layer,10,activation=tf.nn.sigmoid,trainable=True,kernel_initializer=tf.random_normal_initializer)
class_output = tf.nn.softmax(class_layer)

#class_loss_t = tf.multiply(next_hotlabel,tf.divide(tf.log(class_output),tf.log(2.)))
#class_loss_f = tf.multiply(1-next_hotlabel,tf.divide(tf.log(1-class_output),tf.log(2.)))
#class_loss = -tf.reduce_mean(class_loss_t+class_loss_f,axis=1)
class_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=next_label,logits=tf.log(class_output)))
#class_loss = tf.losses.softmax_cross_entropy(next_hotlabel,class_output)

### caluclating and applying gradients
#loss = tf.reduce_mean(generation_loss + tf.multiply(latent_loss,0.01))
loss =generation_loss + tf.multiply(latent_loss,0.0001)
opt = tf.train.AdamOptimizer(0.0005)
gradients0, variables = zip(*opt.compute_gradients(loss))
gradients, _ = tf.clip_by_global_norm(gradients0, 1.)
train_step = opt.apply_gradients(zip(gradients, variables))
#train_step = tf.train.AdamOptimizer(0.0001).minimize(generation_loss)
#train_step1 = tf.train.AdamOptimizer(0.0001).minimize(latent_loss)
train_step2 = tf.train.AdamOptimizer().minimize(tf.multiply(class_loss,1.01))
train_step3 = tf.train.RMSPropOptimizer(0.0002, decay=6e-8).minimize(tf.multiply(loss_desc,0.00001))

###initializing the session
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

###handle iterators manager
handle_train, handle_val = sess.run([iter_train_handle, iter_val_handle])

###initialize loss lists
loss_tr =[]
loss_tr_batch =[]
loss_tr_gen_batch =[]
loss_tr_latent_batch =[]
loss_tst =[]
loss_tst_gen =[]
loss_tst_latent =[]

###training
best_loss = np.inf
saver = tf.train.Saver()
best_epoch=-1
try:
   for epoch in range(epochs):
      for batch in range(training_size/batch_size):
          class_loss0 = np.nan
          tr_loss= np.nan
          tr_descr= np.nan
          out= np.nan
          tr_latent_loss= np.nan
          tr_generation_loss = np.nan
          current_batch = sess.run(next_batch,feed_dict={handle: handle_train})
          _,class_loss0 = sess.run([train_step2,class_loss], feed_dict={next_audio:current_batch["audio"],next_label:current_batch["label"],istrainable:True})
          #class_loss0 = sess.run([class_loss], feed_dict={next_audio:current_batch["audio"],next_label:current_batch["label"]})
          #_, tr_descr= sess.run(
          #         [train_step3,loss_desc],
          #         feed_dict={next_audio: current_batch["audio"],istrainable:False})
          _, tr_loss, out, tr_latent_loss,tr_generation_loss= sess.run(
                   [train_step,loss,output_layer,latent_loss,generation_loss],
                   feed_dict={next_audio: current_batch["audio"],istrainable:True})
          string1=" cl: %.5f, de: %.5f\r" %(np.mean(class_loss0),np.mean(tr_descr))
          #string1=" cl: %.5f" %np.mean(class_loss0)
          string0 = "epoch: %d/%d, batch:  %d/%d, loss: %.3f, gn: %.5f, lt: %.2f"%(epoch+1,
                  epochs,batch+1,training_size/batch_size,
                  np.mean(tr_loss),np.mean(tr_generation_loss)/np.float(length),
                  np.mean(tr_latent_loss))
          sys.stdout.write(string0+string1+"\r")
          sys.stdout.flush()
          loss_tr_batch.append(np.mean(tr_loss))
          loss_tr_gen_batch.append(np.mean(tr_generation_loss))
          loss_tr_latent_batch.append(np.mean(tr_latent_loss))
          #encoded = sess.run(encoded_layer,feed_dict={next_audio:audio_eval})
          if batch+1==training_size/batch_size:
             write(dir_check+str(epoch)+"epoch_"+str(current_batch["label"][0])+".wav",fs,np.reshape(out[0],(length,1)))
             #sys.stdout.write(string0 + string1)
             sys.stdout.write("\n")
          else:
             sys.stdout.write("\r")
          sys.stdout.flush()
      loss_tr.append(np.mean(tr_loss))
      _, tst_loss, tst_latent_loss,tst_generation_loss= sess.run(
                 [train_step, loss,latent_loss,generation_loss],
                 feed_dict={handle: handle_val})
      sys.stdout.write("******eval: loss: %.5f, generative: %.5f, latent: %.5f\n"
                 %(np.mean(tst_loss),np.mean(tst_generation_loss)/np.float(length),
                 np.mean(tst_latent_loss)))
      sys.stdout.flush()
      loss_tst.append(np.mean(tst_loss))
      loss_tst_gen.append(np.mean(tst_generation_loss))
      loss_tst_latent.append(np.mean(tst_latent_loss))
      if np.mean(tr_loss)<best_loss:
         best_loss = np.mean(tr_loss)
         saver.save(sess, dir_check + "best_model.ckpt")
         best_epoch = epoch
      else:
         saver.save(sess, checkpoint_path)

except KeyboardInterrupt:
   saver.save(sess, checkpoint_path)
   np.savetxt(dir_output+"loss_tr.txt",loss_tr,fmt='%.5f')
   np.savetxt(dir_output+"loss_tr_batch.txt",loss_tr_batch,fmt='%.5f')
   np.savetxt(dir_output+"loss_tr_gen_batch.txt",loss_tr,fmt='%.5f')
   np.savetxt(dir_output+"loss_tr_latent_batch.txt",loss_tr,fmt='%.5f')
   np.savetxt(dir_output+"loss_tst.txt",loss_tst,fmt='%.5f')
   np.savetxt(dir_output+"loss_tst_latent.txt",loss_tst_latent,fmt='%.5f')
   np.savetxt(dir_output+"loss_tst_gen.txt",loss_tst_gen,fmt='%.5f')
   text_file = open(dir_output+"info.txt", "w")
   text_file.write("n_hidden: %d\nbest_epoch: %d\nvar: %f" %(n_hidden, best_epoch,sigma2 ))
   text_file.close()
   np.savetxt(dir_output+"model_info.txt",loss_tst_gen,fmt='%.5f')
   indx=[]
   print(audio_eval.shape)
   for i in range(10):
       indx.append(np.where(np.array(label_eval)==i))
   encoded = sess.run(encoded_layer,feed_dict={next_audio:audio_eval})
   fig = plt.figure()
   ax = fig.add_subplot(111, projection='3d')
   if encoded.shape[1]==3:
     for i in range(10):
        ax.plot(encoded[i*50:(i)*50+10,0],encoded[i*50:(i)*50+10,1],
                encoded[i*50:(i)*50+10,2],marker="$ {} $".format(str(i)))
     plt.savefig(dir_output+'scatter.png')
   else:
     from sklearn.decomposition import PCA
     pca = PCA(n_components=3)
     X = pca.fit_transform(encoded)
     for i in range(10):
        ax.plot(X[indx[i],0][0],X[indx[i],1][0],X[indx[i],2][0],
                marker="$ {} $".format(str(label_eval[indx[i][0][0]])))
     plt.savefig(dir_output+'scatter.png')
   mean = []
   for i in range(10):
    print("dist: %.5f" %
          np.sqrt(np.sum(np.square(np.mean(encoded[indx[0]],axis=0)-
          np.mean(encoded[indx[i]],axis=0)))))
   for i in range(10):
      '''print("mean: %s, var: %s" %(str(np.mean(encoded[i*50:(i+1)*50],axis=0)), str(np.var(encoded[i*50:(i+1)*50],axis=0))))
      mean.append(np.mean(encoded[i*50:(i+1)*50],axis=0))
      '''
      output = sess.run(output_layer,feed_dict={
               encoded_layer:np.reshape(np.mean(encoded[indx[i]],axis=0),(1,n_hidden))})
      write(dir_output+"mean"+str(i)+".wav",fs,np.reshape(output,(length,1)))
   np.savetxt(dir_output+"mean.txt",mean,fmt='%.5f')
   sys.exit()

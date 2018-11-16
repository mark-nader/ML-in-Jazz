import torch 
import NeuralNetUtils as nnu

learningRate=0.01
numEpochs=50000

cuda=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

modelD=nnu.NoteDiscriminator(2,100,4,cuda)
modelD=modelD.to(cuda)
modelG=nnu.NoteGenerator(2,20,2,2,0.1,cuda)
modelG=modelG.to(cuda)

criterion=torch.nn.BCELoss()

optimizerD=torch.optim.SGD(modelD.parameters(),lr=learningRate,weight_decay=0.005)
optimizerG=torch.optim.SGD(modelG.parameters(),lr=learningRate,weight_decay=0.005)

real1In=[[1,0,0,0,0,0,0,0,0,0,0,0],
		 [0,1,0,0,0,0,0,0,0,0,0,0]]
					  
real2In=[[0,0,0,1,0,0,0,0,0,0,0,0],
		 [0,0,0,0,1,0,0,0,0,0,0,0]]
		 
realInputD=[]
for i in range(2):
	realInputD.append(torch.tensor([real1In[i],real2In[i]]).to(cuda))
realLabels=torch.tensor([[1],[1]]).float().to(cuda)
fakeLabels=torch.tensor([[0],[0]]).float().to(cuda)

for epoch in range(numEpochs):

	# train D

	# first with real batch

	modelD.zero_grad()
	output=modelD(realInputD)
	errRealD=criterion(output,realLabels)
	errRealD.backward()

	# then with fake batch

	noise=torch.randn(2,2).to(cuda)
	fakeInputD=modelG(noise)
	output=modelD([fI.detach() for fI in fakeInputD])
	errFakeD=criterion(output,fakeLabels)
	errFakeD.backward()
	errD=errRealD+errFakeD
	optimizerD.step()

	# train G
	
	modelG.zero_grad()
	output=modelD(fakeInputD)
	errG=criterion(output,realLabels)
	errG.backward()
	optimizerG.step()
	
	if (epoch+1) % 100 == 0:
		print("Error of Discriminator: {}    |    Error of Generator: {}    -- epoch {}/{}".format(errD,errG,epoch+1,numEpochs))
		
output=modelD(realInputD)
errRealD=criterion(output,realLabels)
print("REAL")
print(realInputD)
print(output)
print(errRealD)
print("====")
print("FAKE")
noise=torch.randn(2,2).to(cuda)
fakeInputD=modelG(noise)
output=modelD(fakeInputD)
errG=criterion(output,realLabels)
print(fakeInputD)
print(output)
print(errG)
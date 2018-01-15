with tf.Graph().as_default():
    if OVERIDE_FOLDER == None:
        # Get the base graph
        if (Net_or_VGG == "Net"):
            TensorBoard_Folder_train = './{4}/{0}_{6}_{1}_n{2}_lr{3}_conv1FS{5}_m{7}_hcv1{8}_sw{9}_TRAIN'.format(
                Net_or_VGG, name, limit, Learning_Rate, folder, Conv1_filtersize, TASK, music_only, conv1_times_hanning,
                sw)
            TensorBoard_Folder_test = './{4}/{0}_{6}_{1}_n{2}_lr{3}_conv1FS{5}_m{7}_hcv1{8}_sw{9}_TEST'.format(
                Net_or_VGG, name, limit, Learning_Rate, folder, Conv1_filtersize, TASK, music_only, conv1_times_hanning,
                sw)
            Saver_Folder = '/{0}_{5}_{1}_n{2}_lr{3}_conv1FS{4}_m{6}_hcv1{7}_sw{8}'.format(Net_or_VGG, name, limit,
                                                                                          Learning_Rate,
                                                                                          Conv1_filtersize, TASK,
                                                                                          music_only,
                                                                                          conv1_times_hanning, sw)
            nets = Gen_audiosetNet(COCHLEAGRAM_LENGTH, numlabels, train_mean_coch, Conv1_filtersize, padding,
                                   poolmethod, conv1_times_hanning, Saved_Weights)
        else:
            TensorBoard_Folder_train = './{4}/{0}_{4}_{1}_n{2}_lr{3}_TRAIN'.format(Net_or_VGG, name, limit,
                                                                                   Learning_Rate, folder, TASK)
            TensorBoard_Folder_test = './{4}/{0}_{4}_{1}_n{2}_lr{3}_TEST'.format(Net_or_VGG, name, limit, Learning_Rate,
                                                                                 folder, TASK)
            Saver_Folder = '/{0}_{4}_{1}_n{2}_lr{3}'.format(Net_or_VGG, name, limit, Learning_Rate, TASK)
            nets = Gen_VGG(COCHLEAGRAM_LENGTH, numlabels, train_mean_coch)
    else:
        TensorBoard_Folder_train = './{0}/{1}_TRAIN'.format(folder, OVERIDE_FOLDER)
        TensorBoard_Folder_test = './{0}/{1}_TEST'.format(folder, OVERIDE_FOLDER)
        Saver_Folder = '/{0}'.format(OVERIDE_FOLDER)
        if (Net_or_VGG == "Net"):
            nets = Gen_audiosetNet(COCHLEAGRAM_LENGTH, numlabels, train_mean_coch, Conv1_filtersize, padding,
                                   poolmethod, conv1_times_hanning, Saved_Weights)
        else:
            nets = Gen_VGG(COCHLEAGRAM_LENGTH, numlabels, train_mean_coch)

    # Get the loss functions
    Cross_Entropy_Train_on_Labels(nets, numlabels, Learning_Rate, multiple_labels)

    merged = tf.summary.merge_all()

    # does all the training
    if (unbalanced):
        train2(test_coch, testlabels, testsize, Saver_Folder, TensorBoard_Folder_train, TensorBoard_Folder_test,
               batchsize, Num_Epochs, nets, trainsize, train_coch, trainlabels, COCHLEAGRAM_LENGTH, cutoff, TASK=TASK,
               SAVE=SAVE)
    else:
        train1(test_coch, testlabels, testsize, Saver_Folder, TensorBoard_Folder_train, TensorBoard_Folder_test,
               batchsize, Num_Epochs, nets, trainsize, train_coch, trainlabels, COCHLEAGRAM_LENGTH)

print("done")
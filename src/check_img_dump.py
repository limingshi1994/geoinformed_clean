def check_img(sample, model, criterion, config):

    test_batch = 1

    colours = cm.get_cmap('viridis', config['num_classes'])
    cmap = colours(np.linspace(0, 1, config['num_classes']))
    cmap[0, -1] = 0  # Set alpha for label 0 to be 0
    cmap[1:, -1] = 0.3  # Set the other alphas for the labels to be 0.3

    avg_meters = {'loss': AverageMeterBatched(),
                  'iou': AverageMeterBatched(),
                  'acc': AverageMeterBatched()}

    odir = f"{config['output_dir']}/inspect_images"
    if not os.path.exists(odir):
        os.makedirs(odir)

    sat = sample["sat"]
    sat = torch.squeeze(sat, 0)
    gt = sample["gt"]
    valid_mask = sample["valid_mask"]
    gt_masked = gt * valid_mask
    gt_masked = torch.squeeze(gt_masked, 0)

    # split all patch tensors into batch_size numbered chunks
    sat_batches = torch.split(sat, test_batch, dim=0)
    gt_batches = torch.split(gt_masked, test_batch, dim=0)

    imid = 0

    for sat_this, gt_this in zip(sat_batches, gt_batches):

        gt_this = make_one_hot(gt_this, config['num_classes'])

        input = sat_this.cuda()
        target = gt_this.cuda()

        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou = iou_score(outputs[-1], target)
        else:
            output = model(input)
            loss, loss_track = criterion(output, target)
            iou = iou_score(output, target)
            acc = pixel_accuracy(output, target)

        avg_meters['loss'].update(list(loss_track))
        avg_meters['iou'].update(list(iou))
        avg_meters['acc'].update(list(acc))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].report()),
            ('iou', avg_meters['iou'].report()),
            ('acc', avg_meters['acc'].report()),
        ])

        output = torch.softmax(output, dim=1)

        pred = torch.argmax(output, dim=1).cpu().numpy()

        # visualization
        # currentfig = plt.figure(dpi=1200)
        satim = sat_this.cpu().numpy()
        satim = np.squeeze(satim, axis=0)
        satim = np.moveaxis(satim, 0, -1)

        pred = np.squeeze(pred, axis=0)
        ovlypred = cmap[pred.flatten()]
        Ra, Ca = pred.shape[:2]
        ovlypred = ovlypred.reshape((Ra, Ca, -1))

        # # Create the figure and axes
        # fig, ax = plt.subplots()
        #
        # # Plot the background image
        # ax.imshow(satim)
        #
        # # Overlay the second image
        # ax.imshow(ovlypred, cmap='viridis', alpha=0.5)  # Use alpha to control transparency
        #
        # # Remove axis ticks and labels
        # ax.set_xticks([])
        # ax.set_yticks([])

        satfig = plt.figure(dpi=1200)
        satfig = plt.imshow(satim)
        plt.savefig(f'{odir}/satimg_{imid}.png', dpi=1200)
        # plt.show(satfig)

        currentfig = plt.imshow(satim)
        # currentfig = plt.imshow(ovlypred)
        plt.savefig(f'{odir}/merged_{imid}.png', dpi=1200)
        plt.close()
        # plt.show(currentfig)
        imid = imid + 1
    torch.cuda.empty_cache()

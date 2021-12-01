class LoadFromFolder(Dataset):
    def __init__(self, main_dir, transform):
         
        # Set the loading directory
        self.main_dir = main_dir
        
        # Apply specific transformes, if needed.
        self.transform = transform
         
        # List all images in folder and count them
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsorted(all_imgs)
        
    def __len__(self):
        # Return the previously computed number of images
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])

        # Use PIL for image loading
        #image = Image.open(img_loc).convert("RGB")
        image = Image.open(img_loc)

        # Apply the transformations
        tensor_image = self.transform(image)
        image_name = self.total_imgs[idx]
        
        # Return image as tensor
        # Return image id
        return tensor_image, image_name

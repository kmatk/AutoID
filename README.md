# **AutoID**
### Identifying Consumer Demographic Groups Via Facial Images
by Kevin Atkinson

## Business Understanding
Two of the major factors in advertising products to consumers are age and gender. While there are many factors in ascertaining the most optimal product advertisements to serve to an individual, these two to encompass a large portion of the equation. In this project we will attempt to use images of individuals to classify their age and gender to better target ads to individuals. 

For many market segments, the subsets of customers who shop online vs in-store may vary significantly. The demogrpahics of customers who enter a retail location is highly valuable information. Using this information could help optimize highly valuable, highly limited shelf space. It can serve to inform decisions regarding end-cap and floor displays. It can also be used to serve targeted ads directly to consumers as they browse shelves for products. 

One such application for this technology would be a cosmetics retail store such as an Ulta or Sephora, where the age group highly influences purchasing decisions. 

Implementing this technology can serve to optimize advertising dollars, retail shelf space, as well as enable more granular inventory selection from location to location.

## Data Understanding

Our data is being sourced from the [B3FD](https://github.com/kbesenic/B3FD) dataset. B3FD stands for Biometrically Filtered Famous Figure Dataset. Introduced in the paper [Picking out the bad apples: unsupervised biometric data ﬁltering forreﬁned age estimation](https://link.springer.com/epdf/10.1007/s00371-021-02323-y?sharing_token=z1NicVj4Fy7P340TvNARsPe4RwlQNchNByi7wbcMAY6I9f3BJkfEnl_nOTlEIb8Wo61IlQRlpMJvoIBvErNdzQVjHI_iw8GtkfEtU2GkEZUAH1OPj6rD6vzQM6L0QxHaTNktc-rMcuc7CpaKb-DYU5QZuxSyGKUtAzk9EUTpuwo=), the data consists of over 375,000 images of famous individuals derived from the [IMDB-WIKI](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) and [CACD](https://bcsiriuschen.github.io/CARC/) datasets. The subjects of the images are 53,759 unique individuals, ranging in age from 0 to 101 years old.

We will be using subsets that have the classification targets of Age and Gender. This is because these are the minimum viable details in order to have reasonably successful targeted advertising. These images have been pre-aligned, and cropped to contain 50% context, as well as resized to be 256x256. You can download the dataset here: images: [B3FD_images.tar.gz](https://ferhr-my.sharepoint.com/:u:/g/personal/kbr122017_fer_hr/EU4lr6xf_ZhBi9vN_i8h_XEByhasE-qqKlcC7iqk5K9XtQ?e=Yox63W), metadata: [B3FD_metadata.tar.gz](https://ferhr-my.sharepoint.com/:u:/g/personal/kbr122017_fer_hr/EcKiZtbTTb5Ep-fN32wCx4oBIcY64Wr8JhxlgPkV33M7cg?e=Q6NtUX).

Because these images have already been cropped cropped to contain 50%+ context (read: face), in its current form this project will be unsuitable for real-world deployment. Prior to deployment we will require an additional step of locating faces within images, cropping the images to the facial bounding box, and then passing that cropped image to our model. That said, what we create will be a vital cog in the gearbox of our AutoID system.
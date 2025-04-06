# mie1517_project_CartoonGAN

This project addresses the challenge of generating cartoon-style images from real-world photographs using Generative Adversarial Networks (GANs). The main goal is to bridge the stylistic gap between realistic visuals and the expressive, stylized nature of cartoons—a crucial step as animation gains prominence across film, advertising, gaming, and social media. A notable example is Ne Zha 2, which grossed over $1.7 billion globally, underscoring the growing influence and market potential of animated content.

Specific main report is in the Group7_Final_Report.html file.

---

## Project Structure

Group7_Final_Report.ipynb: Full project report (in jupyter notebook)

Group7_Final_Report.html: Full project report (in html file)

Group 7 Presentation_Final.pptx: Final presentation slides

app/: PyTorch-based application that transforms regular photos and videos into cartoon style.

dataset/: Dataset folder for training/testing

best_checkpoint.pth: Best model for inference


## Reference

Article for CartoonGAN: https://tobiassunderdiek.github.io/cartoon-gan/#ref_1

Transfer model: https://github.com/TobiasSunderdiek/cartoon-gan/releases

### Testing dataset

Starry Sky
https://www.google.com/search?q=%E6%98%9F%E7%A9%BA&sca_esv=d81613467fe5f619&rlz=1C5CHFA_enCA866CA866&udm=2&biw=1296&bih=634&sxsrf=AHTn8zrgokE5cJ0bfRs0J-94wJpXwHoOYg%3A1742886311022&ei=p1XiZ5iWAcn-ptQPx9DDmQQ&ved=0ahUKEwjYhc-71aSMAxVJv4kEHUfoMEMQ4dUDCBE&uact=5&oq=%E6%98%9F%E7%A9%BA&gs_lp=EgNpbWciBuaYn-epujIFEAAYgAQyBRAAGIAEMgUQABiABDIFEAAYgAQyBRAAGIAEMgUQABiABDIFEAAYgAQyBRAAGIAEMgUQABiABDIFEAAYgARI6SNQzAhYkiJwBXgAkAEAmAFfoAGpBqoBAjEwuAEDyAEA-AEBmAIOoAKrBqgCCsICChAjGCcYyQIY6gLCAgsQABiABBixAxiDAcICCBAAGIAEGLEDwgIOEAAYgAQYsQMYgwEYigXCAg0QABiABBixAxiDARgKwgIKEAAYgAQYsQMYCsICBxAAGIAEGArCAgQQABgDmAMNkgcEMTMuMaAH6CmyBwM4LjG4B40G&sclient=img


Valley
https://www.google.com/search?q=%E5%B1%B1%E8%B0%B7&sca_esv=d81613467fe5f619&rlz=1C5CHFA_enCA866CA866&udm=2&biw=1296&bih=634&sxsrf=AHTn8zric7OhB0sIvtCG1sDNoGTOfd6TeA%3A1742886508012&ei=bFbiZ6dGoKem1A-jz6GYBQ&ved=0ahUKEwinrMaZ1qSMAxWgk4kEHaNnCFMQ4dUDCBE&uact=5&oq=%E5%B1%B1%E8%B0%B7&gs_lp=EgNpbWciBuWxseiwtzIFEAAYgAQyBRAAGIAEMgUQABiABDIFEAAYgAQyBRAAGIAEMgUQABiABDIFEAAYgAQyBRAAGIAEMgUQABiABDIFEAAYgARI1RBQ2AdYiw9wAXgAkAEAmAG1AaAB9gWqAQMzLjS4AQPIAQD4AQGYAgegAq8FqAIKwgIKECMYJxjJAhjqAsICChAAGIAEGEMYigXCAg4QABiABBixAxiDARiKBcICCBAAGIAEGLEDwgILEAAYgAQYsQMYgwGYAwaSBwMzLjSgB7UfsgcDMi40uAepBQ&sclient=img

Countryside
https://www.google.com/search?q=%E4%B9%A1%E6%9D%91&sca_esv=d81613467fe5f619&rlz=1C5CHFA_enCA866CA866&udm=2&biw=1296&bih=634&sxsrf=AHTn8zqp4DEmZsNJDY3rMAHRBVzqYUtk2Q%3A1742886298811&ei=mlXiZ8qfMbuY5OMP45OUmAU&ved=0ahUKEwjK1OW11aSMAxU7DHkGHeMJBVMQ4dUDCBE&uact=5&oq=%E4%B9%A1%E6%9D%91&gs_lp=EgNpbWciBuS5oeadkTIFEAAYgAQyBRAAGIAEMgUQABiABDIFEAAYgAQyBRAAGIAEMgQQABgeMgQQABgeMgQQABgeMgQQABgeMgYQABgFGB5I4BJQ5gRYoBJwAngAkAEAmAGMAaABgAeqAQM1LjS4AQPIAQD4AQGYAgugAr4HqAIKwgIGEAAYBxgewgIKECMYJxjJAhjqAsICCxAAGIAEGLEDGIMBwgIIEAAYgAQYsQOYAweIBgGSBwM1LjagB7knsgcDMy42uAexBw&sclient=img

Landscape
https://www.google.com/search?q=%E9%A3%8E%E6%99%AF&sca_esv=d81613467fe5f619&rlz=1C5CHFA_enCA866CA866&udm=2&biw=1296&bih=634&sxsrf=AHTn8zr227z7QFz9DUfNdN-2KWcqMEDAbw%3A1742886253736&ei=bVXiZ8_dLM-gptQP0PbJiQU&ved=0ahUKEwiPyKag1aSMAxVPkIkEHVB7MlEQ4dUDCBE&uact=5&oq=%E9%A3%8E%E6%99%AF&gs_lp=EgNpbWciBumjjuaZrzIFEAAYgAQyBRAAGIAEMgUQABiABDIFEAAYgAQyBRAAGIAEMgUQABiABDIFEAAYgAQyBRAAGIAEMgUQABiABDIFEAAYgARImhJQigVYvRFwAngAkAEAmAFqoAG0BKoBAzUuMrgBA8gBAPgBAZgCCaAC6wSoAgrCAgcQIxgnGMkCwgIGEAAYBxgewgIKECMYJxjJAhjqAsICCxAAGIAEGLEDGIMBwgIOEAAYgAQYsQMYgwEYigXCAggQABiABBixA5gDBogGAZIHAzcuMqAHsB6yBwM1LjK4B98E&sclient=img

City
https://www.google.com/search?q=%E5%9F%8E%E5%B8%82&sca_esv=d81613467fe5f619&rlz=1C5CHFA_enCA866CA866&udm=2&biw=1296&bih=634&sxsrf=AHTn8zr_WU2ltp_obMgwQ_rFIYw2qTzrOg%3A1742886069599&ei=tVTiZ-GsJN6JptQPzKax6AQ&ved=0ahUKEwih27_I1KSMAxXehIkEHUxTDE0Q4dUDCBE&uact=5&oq=%E5%9F%8E%E5%B8%82&gs_lp=EgNpbWciBuWfjuW4gjIHECMYJxjJAjIFEAAYgAQyBRAAGIAEMgUQABiABDIFEAAYgAQyBRAAGIAEMgUQABiABDIFEAAYgAQyBRAAGIAEMgUQABiABEikI1D_BljPIXAGeACQAQCYAbYBoAHSCKoBAzYuNLgBA8gBAPgBAZgCEKACmgqoAgrCAgoQIxgnGMkCGOoCwgIKEAAYgAQYQxiKBcICCxAAGIAEGLEDGIMBwgIIEAAYgAQYsQPCAg4QABiABBixAxiDARiKBcICBxAAGIAEGArCAgoQABiABBixAxgKmAMWkgcFOS42LjGgB4MwsgcFMy42LjG4B-gJ&sclient=img#vhid=5L_WbPtqgN9IZM&vssid=mosaic

Warm Nature
https://www.google.com/search?q=%E6%B8%A9%E6%9A%96%E8%87%AA%E7%84%B6&sca_esv=d81613467fe5f619&rlz=1C5CHFA_enCA866CA866&udm=2&biw=1296&bih=634&sxsrf=AHTn8zoD6m3mn554y7addV7SGGKNsd9D5w%3A1742886005934&ei=dVTiZ-HkOLKHptQPqv6nsQU&ved=0ahUKEwih85Gq1KSMAxWyg4kEHSr_KVYQ4dUDCBE&uact=5&oq=%E6%B8%A9%E6%9A%96%E8%87%AA%E7%84%B6&gs_lp=EgNpbWciDOa4qeaaluiHqueEtjIHECMYJxjJAki4kAFQwXNY844BcAZ4AJABAJgBfaABnwaqAQM3LjK4AQPIAQD4AQGYAg2gAqoFwgIGEAAYBxgewgIFEAAYgATCAgYQABgEGB6YAwCIBgGSBwQxMS4yoAfIDLIHAzUuMrgHkAU&sclient=img



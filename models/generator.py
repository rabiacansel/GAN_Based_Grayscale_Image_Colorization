import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def down(in_c, out_c, bn=True):
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1)]
            if bn:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        def up(in_c, out_c, dropout=False):
            layers = [nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
                      nn.BatchNorm2d(out_c),
                      nn.ReLU(inplace=True)]
            if dropout:
                layers.append(nn.Dropout(0.5))
            return nn.Sequential(*layers)

    
        self.down1 = down(1, 64, bn=False)     # -> 512
        self.down2 = down(64, 128)             # -> 256
        self.down3 = down(128, 256)            # -> 128
        self.down4 = down(256, 512)            # -> 64
        self.down5 = down(512, 512)            # -> 32
        self.down6 = down(512, 512)            # -> 16
        self.down7 = down(512, 512)            # -> 8

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, 1),      # -> 4
            nn.ReLU(inplace=True)
        )

        self.up1 = up(512, 512, dropout=True)              # -> 8
        self.up2 = up(1024, 512, dropout=True)             # -> 16
        self.up3 = up(1024, 512, dropout=True)             # -> 32
        self.up4 = up(1024, 512)                           # -> 64
        self.up5 = up(1024, 256)                           # -> 128
        self.up6 = up(512, 128)                            # -> 256
        self.up7 = up(256, 64)                             # -> 512

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, 2, 1),            # -> 1024
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)   # 64
        d2 = self.down2(d1)  # 128
        d3 = self.down3(d2)  # 256
        d4 = self.down4(d3)  # 512
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)

        b = self.bottleneck(d7)

        u1 = self.up1(b)
        u2 = self.up2(torch.cat([u1, d7], dim=1))
        u3 = self.up3(torch.cat([u2, d6], dim=1))
        u4 = self.up4(torch.cat([u3, d5], dim=1))
        u5 = self.up5(torch.cat([u4, d4], dim=1))
        u6 = self.up6(torch.cat([u5, d3], dim=1))
        u7 = self.up7(torch.cat([u6, d2], dim=1))

        output = self.final(torch.cat([u7, d1], dim=1))  
        return output

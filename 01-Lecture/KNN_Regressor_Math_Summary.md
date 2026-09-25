# KNN Regressor: กระบวนการคิดทางคณิตศาสตร์

สรุปจากเซกชัน `KNN Regressor` ใน `Linear_Regression.ipynb`

```python
k = 1
j = []
t = np.arange(0, 1, 0.01)

for i in t:
    d = np.abs(X - i)
    d_s = np.argsort(d)[:k]
    j.append(np.mean(Y[d_s]))
```

## 1. ตั้งปัญหา

มีข้อมูลฝึก (training data) $\{(X_n, Y_n)\}_{n=1}^{N}$ โจทย์คือทำนายค่า $\hat{y}$ ที่จุด query ใด ๆ $x_0$ (ในโค้ดคือค่าใน `t` ที่ไล่ตั้งแต่ 0 ถึง 1 ทีละ 0.01)

ต่างจาก Linear/Polynomial Regression ที่ผ่านมาในโน้ตบุ๊ก ซึ่งหาพารามิเตอร์ $R$ คงที่ตัวเดียวมา fit สมการทั้งชุดข้อมูล (Normal Equation: $R = Y X^T (XX^T)^{-1}$) — KNN **ไม่มีขั้นตอน training/fit พารามิเตอร์เลย** มันคือ **non-parametric, instance-based model** ที่เก็บข้อมูลฝึกไว้ทั้งหมด แล้วค่อยคำนวณคำตอบตอน query แต่ละครั้ง (lazy learning)

## 2. คำนวณระยะทางจาก query point ไปยังทุกจุดข้อมูล

```python
d = np.abs(X - i)
```

$$d_n = |X_n - x_0|, \quad n = 1, \dots, N$$

เนื่องจากข้อมูลเป็น 1 มิติ ระยะทาง L1 กับ Euclidean จึงเท่ากัน ถ้าเป็นหลายมิติ ($X_n \in \mathbb{R}^D$) จะต้องใช้

$$d_n = \lVert X_n - x_0 \rVert_2 = \sqrt{\sum_{p=1}^{D} (X_{n,p} - x_{0,p})^2}$$

## 3. เลือก k เพื่อนบ้านที่ใกล้ที่สุด

```python
d_s = np.argsort(d)[:k]
```

`np.argsort(d)` เรียง index ตามค่า $d_n$ จากน้อยไปมาก แล้วตัดเอา $k$ อันดับแรก ได้เซตดัชนี

$$\mathcal{N}_k(x_0) = \{n_1, \dots, n_k\} \quad \text{โดยที่ } d_{n_1} \le \cdots \le d_{n_k} \le d_m \; \forall m \notin \mathcal{N}_k(x_0)$$

คือกลุ่มจุดข้อมูลที่ "อยู่ใกล้" $x_0$ มากที่สุด $k$ จุด

## 4. รวมคำตอบ (Aggregation) — จุดที่ทำให้เป็น Regression

```python
j.append(np.mean(Y[d_s]))
```

$$\hat{y}(x_0) = \frac{1}{k}\sum_{n \in \mathcal{N}_k(x_0)} Y_n$$

นี่คือสมการหลักของ **KNN Regression**: ทำนายด้วยค่า**เฉลี่ยเลขคณิต**ของ label ที่เป็นตัวเลขจากเพื่อนบ้าน $k$ ตัว

เมื่อ $k=1$ (ค่าที่ใช้ในโค้ด) สมการลดรูปเป็น

$$\hat{y}(x_0) = Y_{n^*}, \quad n^* = \arg\min_n |X_n - x_0|$$

ผลลัพธ์คือกราฟรูป **ฟังก์ชันขั้นบันได (piecewise-constant / step function)** ไม่เรียบต่อเนื่อง และไวต่อ noise มาก เพราะทุกช่วงของ $x_0$ อ้างจากจุดข้อมูลจริงเพียงจุดเดียว ถ้าเพิ่ม $k$ กราฟจะเรียบขึ้นเพราะเฉลี่ยจากจุดมากขึ้น (bias เพิ่ม, variance ลด) แต่ถ้า $k$ มากเกินไปจะ underfit เพราะเข้าใกล้ค่าเฉลี่ยรวมของข้อมูลทั้งหมด

## 5. เปรียบเทียบกับ KNN Classification

โครงสร้างของอัลกอริทึม (คำนวณระยะทาง → เรียง → เลือก $k$ เพื่อนบ้าน) **เหมือนกันทุกขั้นตอน** กับ KNN Classification ความต่างอยู่ที่ **ขั้นตอนที่ 4 (Aggregation)** เท่านั้น:

| ประเด็น | KNN Regression | KNN Classification |
|---|---|---|
| ชนิดของ label $Y_n$ | ค่าตัวเลขต่อเนื่อง (continuous) | ค่าหมวดหมู่ (categorical / discrete class) |
| วิธีรวมคำตอบจาก $k$ เพื่อนบ้าน | **ค่าเฉลี่ย (mean)**: $\hat{y}(x_0) = \frac{1}{k}\sum_{n\in\mathcal{N}_k} Y_n$ | **โหวตเสียงข้างมาก (majority vote)**: $\hat{y}(x_0) = \arg\max_{c} \sum_{n\in\mathcal{N}_k} \mathbb{1}[Y_n = c]$ |
| รูปแบบผลลัพธ์ | ค่าตัวเลข ใด ๆ ก็ได้ในช่วงของข้อมูล | หนึ่งใน class ที่มีอยู่จริงในข้อมูลฝึกเท่านั้น |
| ลักษณะกราฟ/ขอบเขต | เส้น/ฟังก์ชันขั้นบันไดของค่าตัวเลข (เมื่อ $k=1$) | พื้นที่แบ่งเขต (decision boundary/region) เป็นโซนสีตาม class |
| Loss/Metric ที่ใช้วัด error | เชิงระยะห่างของค่า เช่น MAE, MSE: $\frac{1}{N}\sum|Y_n - \hat{y}_n|$ | เชิงถูก/ผิดของหมวดหมู่ เช่น Accuracy, Cross-Entropy |
| ผลของ noise ที่ $k=1$ | ค่าทำนายกระโดดตามค่า $Y$ ของจุดที่ใกล้สุด (step function) | ขอบเขตพื้นที่แบ่ง class หยักไปตามจุดข้อมูลรายจุด (overfit ขอบเขต) |

**สรุปสั้น ๆ**: ทั้งสองแบบใช้ตรรกะ "หาความคล้าย/ความใกล้ (distance) แล้วอาศัยเพื่อนบ้านช่วยตัดสิน" เหมือนกัน แต่ Regression ตอบเป็น **ปริมาณ (quantity)** จึงรวมคำตอบด้วยการ**เฉลี่ย** ส่วน Classification ตอบเป็น **หมวดหมู่ (identity)** จึงรวมคำตอบด้วยการ**โหวต** — ความต่างนี้สืบเนื่องมาจากธรรมชาติของ output space: regression มี metric ระยะห่างที่มีความหมาย (เฉลี่ยได้) ในขณะที่ classification มี label เป็น nominal scale ที่เฉลี่ยไม่ได้ ต้องนับความถี่แทน

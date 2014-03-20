#ifndef VEHICLESIM_SLIDEROPTION
#define VEHICLESIM_SLIDEROPTION

#include <Gwen/Controls/HorizontalSlider.h>

#include "../../misc/GetterSetter.h"
#include "../../misc/Updatable.h"

namespace vlr
{
	template <typename baseType>
	class SliderOption
		: public Gwen::Event::Handler, public Updatable, public OptionBase<baseType>
	{
	public:
		typedef float type;

		typedef GetterSetter<baseType, type> GetterType;

		typedef type (baseType::*getterPointerType)() const;
		typedef void (baseType::*setterPointerType)(type);

		SliderOption(Gwen::Controls::Base* parent,
			GetterType getter)
			: _parent(parent), _getter(getter), OptionBase(getter.getBase())
		{
			init();
		}

		SliderOption(Gwen::Controls::Base* parent,
			baseType* base, typename GetterType::finalPointerType pval)
			: _parent(parent), _getter(base, pval), OptionBase(base)
		{
			init();
		}

		SliderOption(Gwen::Controls::Base* parent, baseType* base,
			getterPointerType getterPointer, setterPointerType setterPointer)
			: _parent(parent), _getter(base, getterPointer, setterPointer), OptionBase(base)
		{
			init();
		}

		void setGetter(baseType* base, type* pval)
		{
			_getter = GetterSetter(base, pval);
		}

		void setGetter(baseType* type, getterPointerType getterPointer,
			setterPointerType setterPointer)
		{
			_getter = GetterSetter(base, getterPointer, setterPointer);
		}

		void update() override
		{
			if (!_enabled)
				return;

			_getter.setBase(_base);

			if (_base == nullptr)
				return;
			
			// Update textbox with value
			if (_getter.getValue() != _slider->GetFloatValue())
				_slider->SetFloatValue(_getter.getValue());
		}

		void init()
		{
			_slider = new Gwen::Controls::HorizontalSlider(_parent);
			_slider->onValueChanged.Add(this, &SliderOption::valueChanged);
		}

		Gwen::Controls::Slider* getSlider()
		{
			return _slider;
		}

	protected:
		void valueChanged(Gwen::Event::Info info)
		{
			if (!_enabled)
				return;

			_getter.setValue(_slider->GetFloatValue());
		}

	private:
		GetterType _getter;

		Gwen::Controls::Base* _parent;
		Gwen::Controls::Slider* _slider;
	};
}

#endif /* VEHICLESIM_SLIDEROPTION */
